"""Model definition based on Pytorh-Lightning."""
import os
import json
import configargparse
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl

from deepspeed.ops.adam import DeepSpeedCPUAdam

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Config,
    MT5ForConditionalGeneration,
    MT5Config
)

from information_extraction_t5.features.postprocess import (
    group_qas,
    get_highest_probability_window,
    split_compound_labels_and_predictions,
)
from information_extraction_t5.features.sentences import (
    get_clean_answer_from_subanswer
)
from information_extraction_t5.utils.metrics import (
    normalize_answer,
    t5_qa_evaluate,
    compute_exact,
    compute_f1
)
from information_extraction_t5.utils.freeze import freeze_embeds


class QAClassifier(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(vars(hparams))

        model_path_lower = self.hparams.model_name_or_path.lower()
        config_name_lower = self.hparams.config_name.lower() if self.hparams.config_name else ""

        if 'mt5' in model_path_lower or 'mt5' in config_name_lower:
            print(">> Detected MT5 model. Loading with MT5Config and MT5ForConditionalGeneration.")
            config = MT5Config.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None)
            self.model = MT5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path, from_tf=bool(
                ".ckpt" in self.hparams.model_name_or_path), config=config,
                                                                     cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None)
        else:
            print(">> Detected T5 model. Loading with T5Config and T5ForConditionalGeneration.")
            config = T5Config.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None)
            self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path, from_tf=bool(
                ".ckpt" in self.hparams.model_name_or_path), config=config,
                                                                    cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            do_lower_case=self.hparams.do_lower_case, use_fast=False,
            cache_dir=self.hparams.cache_dir if self.hparams.cache_dir else None)

        self.input_max_length = self.hparams.max_seq_length
        freeze_embeds(self.model)

        # 修正预测缓存路径
        output_dir = self.hparams.output_dir if self.hparams.output_dir else "output/results"
        os.makedirs(output_dir, exist_ok=True)
        model_name_suffix = list(filter(None, self.hparams.model_name_or_path.split("/"))).pop()
        self.cache_fname = os.path.join(output_dir, f"cached_predictions_{model_name_suffix}.pkl")

    def forward(self, x):
        return self.model(x)

class LitQA(QAClassifier, pl.LightningModule):
    
    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        return optimizer

    def training_step(self, batch, batch_idx):
        sentences, labels = batch

        sentences_tokens = self.tokenizer.batch_encode_plus(
            sentences, padding=True, truncation=True,
            max_length=self.input_max_length, return_tensors='pt'
        )
        labels = self.tokenizer.batch_encode_plus(
            labels, padding=True, truncation=True,
            max_length=self.input_max_length, return_tensors='pt'
        )

        inputs = {
            "input_ids": sentences_tokens['input_ids'].to(self.device),
            "labels": labels['input_ids'].to(self.device),
            "attention_mask": sentences_tokens['attention_mask'].to(self.device),
        }

        outputs = self.model(**inputs)

        self.log('train_loss', outputs[0], on_step=True, on_epoch=True,
            prog_bar=True, batch_size=len(sentences)
        )
        return {'loss': outputs[0]}

    def validation_step(self, batch, batch_idx):
        sentences, labels, _, _ = batch

        sentences_tokens = self.tokenizer.batch_encode_plus(
            sentences, padding=True, truncation=True,
            max_length=self.input_max_length, return_tensors='pt'
        )

        inputs = {
            "input_ids": sentences_tokens['input_ids'].to(self.device),
            "attention_mask": sentences_tokens['attention_mask'].to(self.device),
            "max_length": self.hparams.max_length,
        }

        outputs = self.model.generate(**inputs)
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'labels': labels, 'preds': predictions}

    def test_step(self, batch, batch_idx):
        sentences, labels, document_ids, typename_ids = batch

        # if we are using cached predictions, is not necessary to run steps again
        if self.hparams.use_cached_predictions and os.path.exists(self.cache_fname):
            return {'labels': [], 'preds': [], 'doc_ids': [], 'tn_ids': [], 'probs': []}

        sentences_tokens = self.tokenizer.batch_encode_plus(
            sentences, padding=True, truncation=True,
            max_length=self.input_max_length, return_tensors='pt'
        )

        # This is handled differently then the others because of conflicts of
        # the previous approach with quantization.
        inputs = {
            "input_ids": sentences_tokens['input_ids'].to(self.device).long(),
            "attention_mask": sentences_tokens['attention_mask'].to(self.device).long(),
            "max_length": self.hparams.max_length,
            "num_beams": self.hparams.num_beams,
            "early_stopping": True,
        }

        outputs = self.model.generate(**inputs)
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # compute probs
        probs = self._compute_probs(sentences, predictions)

        return {
            'labels': labels, 'preds': predictions, 'doc_ids': document_ids,
            'tn_ids': typename_ids, 'probs': probs
        }

    def validation_epoch_end(self, outputs):
        predictions, labels = [], []
        debug_samples = []

        for output in outputs:
            for label, pred in zip(output['labels'], output['preds']):
                predictions.append(pred)
                labels.append(label)
                if len(debug_samples) < 3:
                    debug_samples.append({"label": label, "prediction": pred})

        # 2. 在输出目录中创建 'debug' 子目录
        output_dir = self.hparams.output_dir if self.hparams.output_dir else "."
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        # 3. 获取当前epoch号，并构造一个独一无二的文件名
        epoch_num = self.current_epoch
        debug_filename = f"epoch_{epoch_num}_samples.json"
        debug_output_path = os.path.join(debug_dir, debug_filename)

        # 4. 将这个epoch的调试样本保存到它自己的文件中
        if debug_samples:
            try:
                with open(debug_output_path, 'w', encoding='utf-8') as f:
                    json.dump(debug_samples, f, ensure_ascii=False, indent=4)
                # 打印一个更清晰的保存信息
                print(f"\nSaved {len(debug_samples)} debug samples for epoch {epoch_num} to {debug_output_path}")
            except Exception as e:
                print(f"\nFailed to save debug samples for epoch {epoch_num}: {e}")

        # 我们将json_mode硬编码为True，因为这个项目就是为此设计的
        results = t5_qa_evaluate(labels, predictions, json_mode=True)

        precision = torch.tensor(results.get('precision', 0.0))
        recall = torch.tensor(results.get('recall', 0.0))
        f1 = torch.tensor(results.get('f1', 0.0))
        exact = torch.tensor(results.get('exact', 0.0))
        bleu = torch.tensor(results.get('bleu', 0.0))
        rougeL = torch.tensor(results.get('rougeL', 0.0))

        log = {
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'val_exact': exact,
            'val_bleu': bleu,
            'val_rougeL': rougeL,
        }
        self.log_dict(log, logger=True, prog_bar=True, on_epoch=True)

        # 打印更详细的验证信息
        print(f"\nValidation Metrics: "
              f"Precision: {precision.item():.2f}, "
              f"Recall: {recall.item():.2f}, "
              f"F1: {f1.item():.2f}, "
              f"Exact: {exact.item():.2f}, "
              f"BLEU: {bleu.item():.2f}, "
              f"ROUGE-L: {rougeL.item():.2f}")

    def test_epoch_end(self, outputs):
        # 1. 获取并创建我们指定的输出目录
        output_dir = self.hparams.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"INFO: Final output files will be saved to: {output_dir}")

        # 2. 定义预测结果的缓存文件路径
        base_cache_name = os.path.basename(self.cache_fname)
        prediction_cache_path = os.path.join(output_dir, base_cache_name.replace("cached_", "cached_preds_"))

        # 3. 收集所有批次的输出
        predictions, labels, document_ids, typename_ids, probs = [], [], [], [], []
        window_ids = []  # 初始化

        for output in outputs:
            if 'probs' in output:
                for label, pred, doc_id, tn_id, prob in zip(
                        output['labels'], output['preds'],
                        output['doc_ids'], output['tn_ids'], output['probs']):
                    predictions.append(pred)
                    labels.append(label)
                    document_ids.append(doc_id)
                    typename_ids.append(tn_id)
                    probs.append(prob)
                    window_ids.append(output.get('window_id', 0))

        # 4. 处理预测结果的缓存
        if self.hparams.use_cached_predictions and os.path.exists(prediction_cache_path):
            print(f'Loading predictions from cached file {prediction_cache_path}')
            cached_df = pd.read_pickle(prediction_cache_path)
            labels = cached_df['labels'].tolist()
            predictions = cached_df['predictions'].tolist()
            document_ids = cached_df['document_ids'].tolist()
            typename_ids = cached_df['typename_ids'].tolist()
            probs = cached_df['probs'].tolist()
            window_ids = cached_df['window_ids'].tolist() if 'window_ids' in cached_df.columns else [0] * len(labels)
        else:
            # 我们在这里调用修正后的备份函数
            self._backup_outputs(labels, predictions, document_ids, typename_ids, probs, window_ids,
                                 prediction_cache_path)

        # 5. 后续的处理逻辑
        if self.hparams.get_highestprob_answer:
            (labels, predictions, document_ids, typename_ids, probs, window_ids) = get_highest_probability_window(
                labels, predictions, document_ids, typename_ids, probs, use_fewer_NA=True
            )

        if self.hparams.split_compound_answers:
            (labels, predictions, document_ids, typename_ids, probs, window_ids, _, _, original_idx,
             disjoint_answer_idx_by_doc_class) = split_compound_labels_and_predictions(
                labels, predictions, document_ids, typename_ids, probs, window_ids
            )
        else:
            print(
                'WARNING: We strongly recommend to set --split_compound_answers=True, even for datasets without compound qas.')
            original_idx = list(range(len(labels)))
            disjoint_answer_idx_by_doc_class = {}

        if self.hparams.group_qas:
            qid_dict_by_typenames = group_qas(typename_ids, group_by_typenames=True)
            qid_dict_by_documents = group_qas(document_ids, group_by_typenames=False)
            qid_dict_by_typenames['ORIG'] = original_idx
            qid_dict_by_documents['ORIG'] = original_idx
        else:
            qid_dict_by_typenames = {'ORIG': original_idx}
            qid_dict_by_documents = {'ORIG': original_idx}

        # 6. 保存所有输出文件
        self._save_outputs(
            labels, predictions, document_ids, probs, window_ids, qid_dict_by_typenames,
            outputs_fname=os.path.join(output_dir, 'outputs_by_typenames.txt'),
            document_classes=list(disjoint_answer_idx_by_doc_class.keys())
        )
        self._save_outputs(
            labels, predictions, typename_ids, probs, window_ids, qid_dict_by_documents,
            outputs_fname=os.path.join(output_dir, 'outputs_by_documents.txt'),
            document_classes=list(disjoint_answer_idx_by_doc_class.keys())
        )

        excel_writer_client = pd.ExcelWriter(os.path.join(output_dir, 'outputs_sheet_client.xlsx'))
        all_idx = []
        for document_class, indices in disjoint_answer_idx_by_doc_class.items():
            qid_dict_by_typenames['DISJOINT_' + document_class] = indices
            qid_dict_by_documents['DISJOINT_' + document_class] = indices
            all_idx += indices
            self._save_sheets(labels, predictions, document_ids, typename_ids, probs, document_class, indices,
                              excel_writer_client)
        excel_writer_client.close()

        qid_dict_by_typenames['DISJOINT_ALL'] = all_idx
        qid_dict_by_documents['DISJOINT_ALL'] = all_idx
        self._save_sheets(labels, predictions, document_ids, typename_ids, probs, 'all', all_idx, output_dir=output_dir)

        # 7. 计算并保存最终指标文件
        results_by_typenames = t5_qa_evaluate(labels, predictions, qid_dict=qid_dict_by_typenames, json_mode=True)
        results_by_documents = t5_qa_evaluate(labels, predictions, qid_dict=qid_dict_by_documents, json_mode=True)

        with open(os.path.join(output_dir, 'metrics_by_typenames.json'), 'w') as f:
            json.dump(results_by_typenames, f, indent=4)
        with open(os.path.join(output_dir, 'metrics_by_documents.json'), 'w') as f:
            json.dump(results_by_documents, f, indent=4)

        # 8. 记录最终指标到日志
        final_f1 = torch.tensor(results_by_typenames.get('f1', 0.0))
        final_exact = torch.tensor(results_by_typenames.get('exact', 0.0))
        log = {'test_f1': final_f1, 'test_exact': final_exact}
        self.log_dict(log, logger=True, on_epoch=True)

    @torch.no_grad()
    def _compute_probs(self, sentences, predictions):
        probs = []
        for sentence, prediction in zip(sentences, predictions):
            input_ids = self.tokenizer.encode(sentence, truncation=True,
                                              max_length=self.input_max_length, return_tensors="pt").to(
                self.device).long()
            output_ids = self.tokenizer.encode(prediction, truncation=True,
                                               max_length=self.input_max_length, return_tensors="pt").to(
                self.device).long()
            if output_ids.shape[1] == 0:  # Handle empty predictions
                probs.append(0.0)
                continue
            outputs = self.model(input_ids=input_ids, labels=output_ids)

            loss = outputs[0]
            # --- 核心修改在这里 ---
            # .item() 直接得到浮点数，然后我们用 np.exp 计算概率
            prob_item = torch.exp(-loss).item()
            probs.append(np.exp(prob_item))  # 使用 numpy.exp 是更稳健的做法

        return probs

    def _backup_outputs(self, labels, predictions, document_ids, typename_ids, probs, window_ids, backup_path):
        """
        修正后的备份函数，它能：
        1. 接收一个明确的备份路径 `backup_path`。
        2. 将 `window_ids` 也包含进备份中。
        3. 使用传入的 `backup_path` 来保存文件。
        """
        print(f"Backing up predictions to: {backup_path}")
        try:
            # 将所有需要备份的数据垂直堆叠起来
            arr = np.vstack([
                np.array(labels, dtype="O"),
                np.array(predictions, dtype="O"),
                np.array(document_ids, dtype="O"),
                np.array(typename_ids, dtype="O"),
                np.array(probs, dtype="O"),
                np.array(window_ids, dtype="O")  # 确保 window_ids 也被备份
            ]).transpose()

            # 创建DataFrame，并指定好列名
            df = pd.DataFrame(arr,
                              columns=['labels', 'predictions', 'document_ids', 'typename_ids', 'probs', 'window_ids'])

            # 使用传入的 backup_path 参数保存文件，而不是硬编码的 self.cache_fname
            df.to_pickle(backup_path)

            print(f"Successfully backed up predictions to {backup_path}")
        except Exception as e:
            print(f"Error during prediction backup: {e}")

    def _save_outputs(self, labels, predictions, doc_or_tn_ids, probs, window_ids, qid_dict=None, outputs_fname='outputs.txt', document_classes=["form"]):
        # This method's logic for writing to a file remains correct, as the fname is now a full path.
        if qid_dict is None: qid_dict = {}
        with open(outputs_fname, 'w') as f:
            f.write('{0:<50} | {1:50} | {2:30} | {3} | {4}\n'.format('label', 'prediction', 'uuid', 'prob', 'window'))
        if qid_dict == {}:
            for label, prediction, doc_or_tn_id, prob, w_id in zip(
                labels, predictions, doc_or_tn_ids, probs, window_ids):
                lab, pred = label, prediction
                if self.hparams.normalize_outputs:
                    lab, pred = normalize_answer(label), normalize_answer(prediction)
                if lab != pred or lab == pred and not self.hparams.only_misprediction_outputs:
                    f.write('{0:<50} | {1:50} | {2:30} | {3} | {4}\n'.format(
                        label, prediction, doc_or_tn_id, prob, w_id))
        else:
            for (kword, list_indices) in qid_dict.items():
                # do not print for ORIG, DISJOINT* and all samples for a specific project/document class
                # those groups are important for metrics, not for outputs visualization
                if kword == 'ORIG' or kword.startswith('DISJOINT') or kword in document_classes: 
                    continue
                f.write(f'===============\n{kword}\n===============\n')
                for idx in list_indices:
                    label, prediction, doc_or_tn_id, prob, w_id = \
                        labels[idx], predictions[idx], doc_or_tn_ids[idx], probs[idx], window_ids[idx]
                    lab, pred = label, prediction
                    if self.hparams.normalize_outputs:
                        lab, pred = normalize_answer(label), normalize_answer(prediction)
                    if lab != pred or lab == pred and not self.hparams.only_misprediction_outputs:
                        f.write('{0:<50} | {1:50} | {2:30} | {3} | {4}\n'.format(
                            label, prediction, doc_or_tn_id, prob, w_id))
        f.close()

    def _save_sheets(self, labels, predictions, document_ids, typename_ids, probs, document_class, indices, writer=None, output_dir="."):
        # Saving disjoint predictions (splitted and clean) in a dataframe
        arr = np.vstack([np.array(document_ids, dtype="O")[indices],
                        np.array(typename_ids, dtype="O")[indices],
                        np.array(labels, dtype="O")[indices],
                        np.array(predictions, dtype="O")[indices],
                        np.array(probs, dtype="O")[indices]]).transpose()
        df = pd.DataFrame(arr, 
            columns=['document_ids', 'typename_ids', 'labels', 'predictions', 'probs']
            ).reset_index(drop=True)

        if document_class == 'all':
            df = df.sort_values(['document_ids', 'typename_ids'])  # hack to keep listing outputs together for each document-class
            df_all_group_doc = df.set_index('document_ids', append=True).swaplevel(0,1)
            # 使用我们传入的output_dir来保存这个特殊的Excel文件
            df_all_group_doc.to_excel(os.path.join(output_dir, 'outputs_sheet.xlsx'))
        else:
            # compute metrics for each pair document_id-typename-id
            df['exact'] = df.apply(lambda x: compute_exact(x['labels'], x['predictions']), axis=1)
            df['f1'] = df.apply(lambda x: compute_f1(x['labels'], x['predictions']), axis=1)

            # remove clue/prefix into brackets
            df['labels'] = df.apply(
                lambda x: ', '.join(get_clean_answer_from_subanswer(x['labels'])),
                axis=1
            )
            df['predictions'] = df.apply(
                lambda x: ', '.join(get_clean_answer_from_subanswer(x['predictions'])),
                axis=1
            )

            # use pivot to get a quadruple of columns (labels, predictions, equal, prob) for each typename
            pivoted = df.pivot(
                index=['document_ids'],
                columns=['typename_ids'],
                values=['labels', 'predictions', 'exact', 'f1', 'probs']
            )
            pivoted = pivoted.swaplevel(0, 1, axis=1).sort_index(axis=1)  # put column (typename_ids) above the values

            # extract typename_ids in the original order (instead of alphanumeric order)
            # get the columns from the document-ids that have more samples
            cols = df[df['document_ids']==df.document_ids.mode()[0]].typename_ids.tolist()
            if len(cols) == len(pivoted.columns) // 5:
                pivoted = pivoted[cols]
            else:
                print('Keeping typenames in alphanumeric order since none of the documents '
                    f'have all the possible qa_ids ({len(cols)} != {len(pivoted.columns) // 5})')

            # save sheet
            pivoted.to_excel(writer, sheet_name=document_class)

    def get_optimizer(self,) -> torch.optim.Optimizer:
        """Define the optimizer"""
        optimizer_name = self.hparams.optimizer
        lr = self.hparams.lr
        weight_decay=self.hparams.weight_decay
        optimizer = getattr(torch.optim, optimizer_name)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        if self.hparams.deepspeed:
            # DeepSpeedCPUAdam provides 5x to 7x speedup over torch.optim.adam(w)
            optimizer = DeepSpeedCPUAdam(
                optimizer_grouped_parameters, lr=lr, 
                weight_decay=weight_decay, eps=1e-4, adamw_mode=True
            )
        else:
            optimizer = optimizer(
                optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay
            )

        print(f'=> Using {optimizer_name} optimizer')

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = configargparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--output_dir", default=".", type=str,
                            help="Path to the directory where final metrics and output files will be saved.")
        parser.add_argument("--model_name_or_path", default='t5-small', type=str, required=True,
                            help="Path to pretrained model or model identifier from huggingface.co/models")
        parser.add_argument("--config_name", default="", type=str,
                            help="Pretrained config name or path if not the same as model_name")
        parser.add_argument("--tokenizer_name", default="", type=str,
                            help="Pretrained tokenizer name or path if not the same as model_name")
        parser.add_argument("--cache_dir", default="", type=str,
                            help="Where do you want to store the pre-trained models downloaded from s3")
        parser.add_argument("--do_lower_case", action="store_true",
                            help="Set this flag if you are using an uncased model.")
        parser.add_argument("--max_seq_length", default=384, type=int,
                            help="The maximum total input sequence length after WordPiece tokenization.")
        parser.add_argument("--max_size", default=1024, type=int,
                            help="The maximum input length after char-based tokenization.")
        parser.add_argument("--max_length", default=120, type=int,
                            help="The maximum total output sequence length generated by the model.")
        parser.add_argument("--num_beams", default=1, type=int,
                            help="Number of beams for beam search. 1 means no beam search.")
        parser.add_argument("--get_highestprob_answer", action="store_true",
                            help="If true, get the answer from the sliding-window that gives highest probability.")
        parser.add_argument("--split_compound_answers", action="store_true",
                             help="If true, split the T5 outputs into individual answers.")
        parser.add_argument("--group_qas", action="store_true",
                            help="If true, use group qas to get individual metrics.")
        parser.add_argument("--only_misprediction_outputs", action="store_true",
                            help="If true, return only mispredictions in the output file.")
        parser.add_argument("--normalize_outputs", action="store_true",
                            help="If true, normalize label and prediction to include in the output file.")
        return parser

# information_extraction_t5/utils/metrics.py (THE FINAL, ULTIMATE, ALL-INCLUSIVE VERSION)
import collections
import json
import re
import string
from typing import Dict, Optional
import unicodedata

from rouge_score import rouge_scorer
import sacrebleu


# --- SECTION 1: 原作者的 SQuAD 风格函数 (完整保留，确保兼容性) ---
def normalize_answer(s):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    return white_space_fix(remove_articles(strip_accents(remove_punc(lower(s)))))


def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks, pred_toks = get_tokens(a_gold), get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ("exact", 100.0 * sum(exact_scores.values()) / total),
            ("f1", 100.0 * sum(f1_scores.values()) / total),
            ("total", total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ("total", total),
        ])


def get_raw_scores(answers, preds):
    exact_scores, f1_scores = {}, {}
    for i, (answer, pred) in enumerate(zip(answers, preds)):
        exact_scores[i] = compute_exact(answer, pred)
        f1_scores[i] = compute_f1(answer, pred)
    return exact_scores, f1_scores


# --- SECTION 1: 辅助函数 ---
def _compute_arg_f1(gold_args, pred_args):
    """为arguments计算token-level的F1分数"""
    gold_toks = " ".join(gold_args.values()).split()
    pred_toks = " ".join(pred_args.values()).split()
    if not gold_toks and not pred_toks: return 1.0
    if not gold_toks or not pred_toks: return 0.0
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def _compute_weighted_score(gold_event, pred_event):
    """计算一个事件的加权内容匹配得分"""
    score = 0.0
    # 权重分配：trigger=0.4, type=0.1, arguments=0.5
    if gold_event.get('trigger') == pred_event.get('trigger'):
        score += 0.4
    if gold_event.get('event_type') == pred_event.get('event_type'):
        score += 0.1
    score += 0.5 * _compute_arg_f1(gold_event.get('arguments', {}), pred_event.get('arguments', {}))
    return score


def _compute_sets_and_maps(answers, preds):
    """一次性从数据中提取所有需要的信息"""
    results = {
        "gold_maps_strict": [], "pred_maps_strict": [],
        "gold_sets_relaxed": [], "pred_sets_relaxed": [],
        "gold_sets_enhanced": [], "pred_sets_enhanced": []
    }
    for answer, pred in zip(answers, preds):
        try:
            gold_data = json.loads(answer) if answer else []
            if not isinstance(gold_data, list): gold_data = []
        except (json.JSONDecodeError, TypeError):
            gold_data = []
        try:
            pred_data = json.loads(pred) if pred else []
            if not isinstance(pred_data, list): pred_data = []
        except (json.JSONDecodeError, TypeError):
            pred_data = []

        # 构造三种模式的集合与映射
        gm_s, pm_s = {}, {};
        gs_r, ps_r = set(), set();
        gs_e, ps_e = set(), set()
        for p in gold_data:
            if isinstance(p.get('cause'), dict) and isinstance(p.get('effect'), dict):
                cause_id, effect_id = p['cause'].get('event_id'), p['effect'].get('event_id')
                if cause_id and effect_id:
                    pair_id = (cause_id, effect_id)
                    gm_s[pair_id] = p
                    gs_r.add(pair_id)
                    gs_e.add((cause_id, effect_id, p['cause'].get('trigger'), p['effect'].get('trigger')))
        for p in pred_data:
            if isinstance(p.get('cause'), dict) and isinstance(p.get('effect'), dict):
                cause_id, effect_id = p['cause'].get('event_id'), p['effect'].get('event_id')
                if cause_id and effect_id:
                    pair_id = (cause_id, effect_id)
                    pm_s[pair_id] = p
                    ps_r.add(pair_id)
                    ps_e.add((cause_id, effect_id, p['cause'].get('trigger'), p['effect'].get('trigger')))

        results["gold_maps_strict"].append(gm_s);
        results["pred_maps_strict"].append(pm_s)
        results["gold_sets_relaxed"].append(gs_r);
        results["pred_sets_relaxed"].append(ps_r)
        results["gold_sets_enhanced"].append(gs_e);
        results["pred_sets_enhanced"].append(ps_e)
    return results


# --- SECTION 2: 最终的、集大成的评估函数 ---
def t5_qa_evaluate(answers, preds, qid_dict: Optional[Dict] = None):
    data = _compute_sets_and_maps(answers, preds)

    # 计算三种F1
    def calculate_metrics(gold_sets, pred_sets):
        tp, fp, fn = 0, 0, 0
        for g, p in zip(gold_sets, pred_sets):
            tp += len(g.intersection(p));
            fp += len(p - g);
            fn += len(g - p)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision * 100, "recall": recall * 100, "f1": f1 * 100}

    relaxed_metrics = calculate_metrics(data["gold_sets_relaxed"], data["pred_sets_relaxed"])
    enhanced_metrics = calculate_metrics(data["gold_sets_enhanced"], data["pred_sets_enhanced"])

    # 计算最严格的加权F1
    strict_tp_score, strict_tp_count = 0, 0
    for gold_map, pred_map in zip(data["gold_maps_strict"], data["pred_maps_strict"]):
        tp_ids = set(gold_map.keys()).intersection(set(pred_map.keys()))
        strict_tp_count += len(tp_ids)
        for pair_id in tp_ids:
            gold_pair, pred_pair = gold_map[pair_id], pred_map[pair_id]
            cause_score = _compute_weighted_score(gold_pair['cause'], pred_pair['cause'])
            effect_score = _compute_weighted_score(gold_pair['effect'], pred_pair['effect'])
            strict_tp_score += (cause_score + effect_score) / 2.0

    total_pred = sum(len(p) for p in data["pred_sets_relaxed"])
    total_gold = sum(len(g) for g in data["gold_sets_relaxed"])

    strict_precision = strict_tp_score / total_pred if total_pred > 0 else 0.0
    strict_recall = strict_tp_score / total_gold if total_gold > 0 else 0.0
    strict_f1 = (2 * strict_precision * strict_recall) / (strict_precision + strict_recall) if (
                                                                                                           strict_precision + strict_recall) > 0 else 0.0

    strict_metrics = {"precision": strict_precision * 100, "recall": strict_recall * 100, "f1": strict_f1 * 100}

    # 组装最终报告
    evaluation = collections.OrderedDict([
        ("relaxed_f1_id_match", relaxed_metrics),
        ("enhanced_f1_id_trigger_match", enhanced_metrics),
        ("strict_f1_weighted_content", strict_metrics),
    ])

    # 计算文本生成指标
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = 0.0, 0.0, 0.0
    for gold, pred in zip(answers, preds):
        s = scorer.score(gold, pred)
        rouge1, rouge2, rougeL = rouge1 + s['rouge1'].fmeasure, rouge2 + s['rouge2'].fmeasure, rougeL + s[
            'rougeL'].fmeasure
    total = len(answers)
    if total > 0:
        evaluation['rouge1'] = (rouge1 / total) * 100
        evaluation['rouge2'] = (rouge2 / total) * 100
        evaluation['rougeL'] = (rougeL / total) * 100
    references = [[gold] for gold in answers]
    bleu_score = sacrebleu.corpus_bleu(preds, references)
    evaluation['bleu'] = bleu_score.score

    return evaluation
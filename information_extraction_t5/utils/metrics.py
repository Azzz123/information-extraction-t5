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


# --- SECTION 2: 我们新增的、更科学的评估逻辑 ---
def make_hashable(obj):
    if isinstance(obj, dict):
        return frozenset((k, make_hashable(v)) for k, v in sorted(obj.items()))
    if isinstance(obj, list):
        return tuple(make_hashable(e) for e in obj)
    return obj


def compute_prf1_sets(gold_sets, pred_sets):
    total_tp, total_fp, total_fn = 0, 0, 0
    for gold_set, pred_set in zip(gold_sets, pred_sets):
        total_tp += len(gold_set.intersection(pred_set))
        total_fp += len(pred_set - gold_set)
        total_fn += len(gold_set - pred_set)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# --- SECTION 3: 最终的、集大成的评估函数 (这是唯一会被我们调用的主函数) ---
def t5_qa_evaluate(answers, preds, qid_dict: Optional[Dict] = None):
    gold_sets_strict, pred_sets_strict = [], []
    gold_sets_relaxed, pred_sets_relaxed = [], []

    for answer, pred in zip(answers, preds):
        # 严格模式
        gold_set_s = set()
        try:
            gold_data = json.loads(answer)
            if isinstance(gold_data, list):
                for item in gold_data: gold_set_s.add(make_hashable(item))
        except (json.JSONDecodeError, TypeError):
            pass
        gold_sets_strict.append(gold_set_s)
        pred_set_s = set()
        try:
            pred_data = json.loads(pred)
            if isinstance(pred_data, list):
                for item in pred_data: pred_set_s.add(make_hashable(item))
        except (json.JSONDecodeError, TypeError):
            pass
        pred_sets_strict.append(pred_set_s)

        # 宽松模式
        gold_set_r = set()
        try:
            gold_data = json.loads(answer)
            if isinstance(gold_data, list):
                for item in gold_data:
                    cause_id = item.get('cause', {}).get('event_id')
                    effect_id = item.get('effect', {}).get('event_id')
                    if cause_id and effect_id: gold_set_r.add((cause_id, effect_id))
        except (json.JSONDecodeError, TypeError):
            pass
        gold_sets_relaxed.append(gold_set_r)
        pred_set_r = set()
        try:
            pred_data = json.loads(pred)
            if isinstance(pred_data, list):
                for item in pred_data:
                    cause_id = item.get('cause', {}).get('event_id')
                    effect_id = item.get('effect', {}).get('event_id')
                    if cause_id and effect_id: pred_set_r.add((cause_id, effect_id))
        except (json.JSONDecodeError, TypeError):
            pass
        pred_sets_relaxed.append(pred_set_r)

    sp, sr, sf1 = compute_prf1_sets(gold_sets_strict, pred_sets_strict)
    rp, rr, rf1 = compute_prf1_sets(gold_sets_relaxed, pred_sets_relaxed)

    evaluation = collections.OrderedDict([
        ("strict_precision", sp * 100), ("strict_recall", sr * 100), ("strict_f1", sf1 * 100),
        ("relaxed_precision", rp * 100), ("relaxed_recall", rr * 100), ("relaxed_f1", rf1 * 100),
    ])

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
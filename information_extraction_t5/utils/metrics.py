# information_extraction_t5/utils/metrics.py (FINAL VERSION)
import collections
import json
import re
import string
from typing import Dict, Optional
import unicodedata

# --- 新增导入 ---
from rouge_score import rouge_scorer
import sacrebleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# --- 原有函数 (保持不变) ---
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
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
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')
    return white_space_fix(remove_articles(strip_accents(remove_punc(lower(s)))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# --- 升级后的JSON评估函数 (返回P, R, F1, Exact) ---
def make_hashable(obj):
    """ 递归地将一个（可能包含列表和字典的）对象转换成可哈希的对象。"""
    if isinstance(obj, dict):
        return frozenset((k, make_hashable(v)) for k, v in sorted(obj.items()))
    if isinstance(obj, list):
        return tuple(make_hashable(e) for e in obj)
    return obj

def compute_prf1_json(a_gold_str, a_pred_str):
    """
    计算两个JSON字符串表示的因果关系对集合的Precision, Recall, F1和Exact Match。
    这个版本能正确处理嵌套的JSON结构。
    """
    gold_set = set()
    try:
        gold_data = json.loads(a_gold_str)
        if isinstance(gold_data, list):
            for item in gold_data:
                gold_set.add(make_hashable(item))
    except (json.JSONDecodeError, TypeError):
        # 如果解析失败或结构无法哈希，gold_set 保持为空
        pass

    pred_set = set()
    try:
        pred_data = json.loads(a_pred_str)
        if isinstance(pred_data, list):
            for item in pred_data:
                pred_set.add(make_hashable(item))
    except (json.JSONDecodeError, TypeError):
        # 如果解析失败或结构无法哈希，pred_set 保持为空
        pass

    true_positives = len(gold_set.intersection(pred_set))
    false_positives = len(pred_set - gold_set)
    false_negatives = len(gold_set - pred_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = 1 if gold_set == pred_set else 0

    return precision, recall, f1, exact_match

# --- 升级后的 get_raw_scores (存储P, R, F1) ---
def get_raw_scores(answers, preds, json_mode=False):
    """根据json_mode选择不同的评估方式计算 P, R, F1, Exact。"""
    exact_scores, f1_scores, precision_scores, recall_scores = {}, {}, {}, {}

    for i, (answer, pred) in enumerate(zip(answers, preds)):
        if json_mode:
            p, r, f1, exact = compute_prf1_json(answer, pred)
            precision_scores[i] = p
            recall_scores[i] = r
            f1_scores[i] = f1
            exact_scores[i] = exact
        else:
            # 传统模式下，P和R与F1相同（因为是基于单个样本）
            f1 = compute_f1(answer, pred)
            precision_scores[i] = f1
            recall_scores[i] = f1
            f1_scores[i] = f1
            exact_scores[i] = compute_exact(answer, pred)

    return precision_scores, recall_scores, f1_scores, exact_scores

# --- 升级后的 make_eval_dict (包含P, R, F1) ---
def make_eval_dict(precision_scores, recall_scores, f1_scores, exact_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ("precision", 100.0 * sum(precision_scores.values()) / total if total > 0 else 0),
            ("recall", 100.0 * sum(recall_scores.values()) / total if total > 0 else 0),
            ("f1", 100.0 * sum(f1_scores.values()) / total if total > 0 else 0),
            ("exact", 100.0 * sum(exact_scores.values()) / total if total > 0 else 0),
            ("total", total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ("precision", 100.0 * sum(precision_scores[k] for k in qid_list) / total if total > 0 else 0),
            ("recall", 100.0 * sum(recall_scores[k] for k in qid_list) / total if total > 0 else 0),
            ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total if total > 0 else 0),
            ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total if total > 0 else 0),
            ("total", total),
        ])

# --- 最终的、集大成的评估函数 (包含所有指标) ---
def t5_qa_evaluate(answers, preds, qid_dict: Optional[Dict] = None, json_mode=False):
    """
    最终评估函数，计算PRF1, Exact, BLEU, ROUGE。
    """
    if qid_dict is None:
        qid_dict = {}

    # 1. 计算核心指标 (P, R, F1, Exact)
    precision_scores, recall_scores, f1_scores, exact_scores = get_raw_scores(answers, preds, json_mode=json_mode)
    evaluation = make_eval_dict(precision_scores, recall_scores, f1_scores, exact_scores)

    # 2. 计算生成指标 (BLEU, ROUGE)
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = 0.0, 0.0, 0.0
    for gold, pred in zip(answers, preds):
        scores = scorer.score(gold, pred)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougeL += scores['rougeL'].fmeasure
    
    total = len(answers)
    if total > 0:
        evaluation['rouge1'] = (rouge1 / total) * 100
        evaluation['rouge2'] = (rouge2 / total) * 100
        evaluation['rougeL'] = (rougeL / total) * 100

    # BLEU (using sacrebleu)
    # sacrebleu expects a list of references for each prediction
    references = [[gold] for gold in answers]
    bleu_score = sacrebleu.corpus_bleu(preds, references)
    evaluation['bleu'] = bleu_score.score

    # 3. 按组计算核心指标
    for (kword, qid_list) in qid_dict.items():
        evaluation[kword] = make_eval_dict(precision_scores, recall_scores, f1_scores, exact_scores, qid_list)

    return evaluation
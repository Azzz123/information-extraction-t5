# scripts/convert_milcause_to_squad_enhanced.py

import json
import os
import argparse
from tqdm import tqdm

def format_candidates(candidate_pairs):
    """
    将候选对列表序列化为对LLM友好的、人类可读的字符串。
    """
    formatted_strings = []
    for i, pair in enumerate(candidate_pairs):
        try:
            cause_trigger = pair['event_1']['trigger']
            effect_trigger = pair['event_2']['trigger']
            # 为了简洁，我们只使用触发词作为事件的代表
            formatted_strings.append(
                f"候选{i+1}: [原因事件: '{cause_trigger}'] -> [结果事件: '{effect_trigger}']"
            )
        except (KeyError, TypeError):
            # 如果结构不完整，跳过这个候选对
            continue
    return " \n".join(formatted_strings)

def convert_milcause_to_squad(input_file, output_file):
    """
    将 MilCause 数据格式转换为 SQuAD 格式。
    采用“宏问题”策略，并将候选对列表序列化后加入到 context 中。
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            milcause_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}. Please check its format.")
        return

    squad_data = {"data": [], "version": "0.1"}

    fixed_question = "请基于上下文和给定的候选对，以JSON列表格式，筛选并输出所有真实存在的因果关系对。"
    fixed_id_typename = "milcause.causal_relations"

    print(f"Starting enhanced conversion for {input_file}...")
    for i, item in enumerate(tqdm(milcause_data, desc="Converting records")):
        try:
            input_content = json.loads(item['input'])
            original_text = input_content['text']
            candidate_pairs = input_content['candidate_pairs']
            answer_text = item['output']
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"\nWarning: Skipping record {i} due to parsing error: {e}")
            continue

        # 序列化候选对列表
        formatted_candidates_text = format_candidates(candidate_pairs)

        # 构建增强后的上下文
        # 使用特殊分隔符 [SEP] 来区分原文和候选对列表
        enhanced_context = f"{original_text}\n\n[SEP]\n\n请从以下候选对中进行筛选：\n{formatted_candidates_text}"

        answer_start = -1

        qas = [{
            "answers": [{
                "answer_start": answer_start,
                "text": answer_text
            }],
            "question": fixed_question,
            "id": fixed_id_typename
        }]

        paragraph = {
            "context": enhanced_context,
            "qas": qas
        }

        document = {
            "title": f"milcause_doc_{i}",
            "paragraphs": [paragraph]
        }

        squad_data["data"].append(document)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(squad_data, f, ensure_ascii=False, indent=2)

    print(f"\nSuccessfully converted {len(squad_data['data'])} records with enhanced context.")
    print(f"Output saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert MilCause JSON to SQuAD format with enhanced context.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input MilCause JSON file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output SQuAD-formatted JSON file.')
    args = parser.parse_args()
    convert_milcause_to_squad(args.input_file, args.output_file)
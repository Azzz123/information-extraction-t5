# scripts/convert_milcause_to_squad_enhanced.py (FINAL VERSION)

import json
import os
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split  # 引入强大的分割工具


def format_candidates(candidate_pairs):
    """将候选对列表序列化为对LLM友好的、人类可读的字符串。"""
    # (此函数保持不变)
    formatted_strings = []
    for i, pair in enumerate(candidate_pairs):
        try:
            cause_trigger = pair['event_1']['trigger']
            effect_trigger = pair['event_2']['trigger']
            formatted_strings.append(
                f"候选{i + 1}: [原因事件: '{cause_trigger}'] -> [结果事件: '{effect_trigger}']"
            )
        except (KeyError, TypeError):
            continue
    return " \n".join(formatted_strings)


def convert_records_to_squad_format(records, dataset_name=""):
    """
    一个核心的转换函数，将一批记录转换为SQuAD格式的data部分。
    """
    squad_data_list = []
    fixed_question = "请基于上下文和给定的候选对，以JSON列表格式，筛选并输出所有真实存在的因果关系对。"
    fixed_id_typename = "milcause.causal_relations"

    for i, item in enumerate(tqdm(records, desc=f"Converting {dataset_name} records")):
        try:
            input_content = json.loads(item['input'])
            original_text = input_content['text']
            candidate_pairs = input_content['candidate_pairs']
            answer_text = item['output']
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"\nWarning: Skipping record in {dataset_name} at index {i} due to parsing error: {e}")
            continue

        enhanced_context = f"{original_text}\n\n[SEP]\n\n请从以下候选对中进行筛选：\n{format_candidates(candidate_pairs)}"
        answer_start = -1

        qas = [{"answers": [{"answer_start": answer_start, "text": answer_text}], "question": fixed_question,
                "id": fixed_id_typename}]
        paragraph = {"context": enhanced_context, "qas": qas}
        document = {"title": f"milcause_{dataset_name}_doc_{i}", "paragraphs": [paragraph]}
        squad_data_list.append(document)

    return squad_data_list


def main():
    parser = argparse.ArgumentParser(description="Convert MilCause JSON to SQuAD format with automated dev split.")
    parser.add_argument('--input_train_file', type=str, required=True,
                        help='Path to the input raw MilCause train JSON file.')
    parser.add_argument('--input_test_file', type=str, required=True,
                        help='Path to the input raw MilCause test JSON file.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the processed SQuAD files (train.json, dev.json, test.json).')
    parser.add_argument('--dev_ratio', type=float, default=0.1,
                        help='Ratio of the training data to be used for the development/validation set.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the train-dev split to ensure reproducibility.')
    args = parser.parse_args()

    # --- 1. 创建输出目录 ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory '{args.output_dir}' is ready.")

    # --- 2. 处理训练集和验证集 ---
    print(f"\nLoading raw training data from '{args.input_train_file}'...")
    with open(args.input_train_file, 'r', encoding='utf-8') as f:
        raw_train_data = json.load(f)

    # 使用sklearn进行安全的、无交叉的分割
    train_records, dev_records = train_test_split(
        raw_train_data,
        test_size=args.dev_ratio,
        random_state=args.seed
    )
    print(
        f"Splitting raw training data: {len(train_records)} for train, {len(dev_records)} for dev (ratio: {args.dev_ratio}).")

    # 分别转换训练和验证记录
    squad_train_data = convert_records_to_squad_format(train_records, "train")
    squad_dev_data = convert_records_to_squad_format(dev_records, "dev")

    # 封装成完整的SQuAD JSON结构并保存
    final_train_json = {'data': squad_train_data, 'version': '0.1'}
    final_dev_json = {'data': squad_dev_data, 'version': '0.1'}

    train_output_path = os.path.join(args.output_dir, 'train.json')
    dev_output_path = os.path.join(args.output_dir, 'dev.json')

    with open(train_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_train_json, f, ensure_ascii=False, indent=2)
    print(f"\nProcessed training data saved to '{train_output_path}'")

    with open(dev_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dev_json, f, ensure_ascii=False, indent=2)
    print(f"Processed development data saved to '{dev_output_path}'")

    # --- 3. 处理测试集 ---
    print(f"\nLoading raw test data from '{args.input_test_file}'...")
    with open(args.input_test_file, 'r', encoding='utf-8') as f:
        raw_test_data = json.load(f)

    squad_test_data = convert_records_to_squad_format(raw_test_data, "test")
    final_test_json = {'data': squad_test_data, 'version': '0.1'}
    test_output_path = os.path.join(args.output_dir, 'test.json')

    with open(test_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_test_json, f, ensure_ascii=False, indent=2)
    print(f"\nProcessed test data saved to '{test_output_path}'")

    print("\nAll data processing is complete!")


if __name__ == '__main__':
    main()
import json
import os
import re
import jsonlines
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests

# 定义zero-shot prompting的提示模板
zero_shot_prompt = '''
下面是一道{question_type}，请先详细分析问题，最后给出选项。
{question}
{option}
'''

def generate_query(data, technique="zero-shot"):
    if technique == "zero-shot":
        prompt = zero_shot_prompt
    else:
        raise ValueError("Unsupported prompting technique")

    question = data['question']
    option = '\n'.join([k+'. '+v for k,v in data['option'].items() if v != ''])
    return prompt.format_map({'question': question, 'option': option, 'question_type': data['question_type']})

def prepare_data(input_path, output_path, technique="zero-shot"):
    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    print(f"len:{len(data)}")
    jsonl_data = [
        {
            "id": id,
            "query": generate_query(item, technique=technique),
            "model_answer": "",
            "question_type": item['question_type'],
            "groundtruth": item['answer']
        }
        for id, item in enumerate(data)
    ]

    with open(output_path, "w", encoding="utf-8") as file:
        for entry in jsonl_data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"Prepare finished, output to '{output_path}'")

def match_choice(text):
    match = re.findall(r'.*?([A-E]+(?:[、, ]+[A-E]+)*)', text)
    return ''.join(re.split(r'[、, ]+', match[-1])) if match else ''

def calculate_score(question_type, groundtruth, model_answer):
    choice = match_choice(model_answer)
    if len(choice) > 1 and question_type != '多项选择题':
        choice = choice[0]
    return choice == groundtruth, choice

def score_result(input_path, wrong_ans_path, score_path):
    with jsonlines.open(input_path, "r") as reader:
        items = list(reader)

    question_type = ['最佳选择题', '配伍选择题', '综合分析选择题', '多项选择题']
    type2score = {q_type: {'correct': 0, 'total': 0} for q_type in question_type}
    wrong_data = []

    for item in items:
        q_type = item['question_type']
        groundtruth = item['groundtruth']
        is_correct, model_choice = calculate_score(q_type, groundtruth, item['model_answer'])

        if is_correct:
            type2score[q_type]['correct'] += 1
        else:
            item['model_choice'] = model_choice
            wrong_data.append(item)

        type2score[q_type]['total'] += 1

    total_correct = sum(item['correct'] for item in type2score.values())
    total_questions = len(items)
    for q_type, item in type2score.items():
        if item['total'] > 0:
            accuracy = item['correct'] / item['total']
            print(f'[{q_type}]准确率：{accuracy:.3f}  题目总数：{item["total"]}')

    print(f'总分：{total_correct}  / 满分：{total_questions}')
    print(f'错误题目：{len(wrong_data)}道，已输出到 {wrong_ans_path}')
    
    with open(wrong_ans_path, 'w', encoding='utf-8') as fw:
        json.dump(wrong_data, fw, ensure_ascii=False, indent=4)

    score_info = {
        'total_score': total_correct,
        'total_questions': total_questions,
        'scores_by_type': type2score
    }
    
    with open(score_path, 'w', encoding='utf-8') as fscore:
        json.dump(score_info, fscore, ensure_ascii=False, indent=4)

def api_call(input_path, output_path, api_key, base_url):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    def process_item(item):
        try:
            response = requests.post(
                f"{base_url}",
                headers=headers,
                json={
                    "model": "deepseek-ai/DeepSeek-V3",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": item["query"]}
                    ]
                }
            )
            response.raise_for_status()
            item["model_answer"] = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error processing item: {item['query']}. Error: {e}")
            item["model_answer"] = "Error"
        return item

    with jsonlines.open(input_path, "r") as reader:
        items_to_process = list(reader)

    with jsonlines.open(output_path, "w") as writer:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_item, item): item for item in items_to_process}
            for future in tqdm(futures, total=len(items_to_process), desc="Processing items"):
                try:
                    writer.write(future.result())
                except Exception as e:
                    print(f"Error writing item: {futures[future]['query']}. Error: {e}")

if __name__ == '__main__':
    # 集成参数
    input_path = "./data/1.exam.json"
    output_path = "./data/zero-shot_prepared.jsonl"
    api_key = "sk-"
    base_url = "https://api.siliconflow.cn/v1/chat/completions"
    after_api_path = "./data/zero-shot_aftgpt.jsonl"
    wrong_ans_path = "./data/zero-shot_wrong_ans.json"
    score_path = "./data/zero-shot_score.json"
    technique = "zero-shot"

    # Step 1: Prepare data
    prepare_data(input_path, output_path, technique=technique)

    # Step 2: API call to get answers
    api_call(output_path, after_api_path, api_key, base_url)

    # Step 3: Score the results
    score_result(after_api_path, wrong_ans_path, score_path)
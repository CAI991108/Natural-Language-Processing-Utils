import json
import os
import re
import jsonlines
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests
import random
import time
from collections import Counter

# 定义chain-of-thought prompting的提示模板
chain_of_thought_prompt = '''
下面是几道{question_type}，请先详细分析问题，最后给出选项。
示例1:
问题: {example_question_1}
选项: {example_option_1}
答案: {example_answer_1}
解析: {example_explanation_1}

示例2:
问题: {example_question_2}
选项: {example_option_2}
答案: {example_answer_2}
解析: {example_explanation_2}

现在请回答以下问题:
问题: {question}
选项: {option}
让我们一步一步地思考。
'''

def generate_query(data, examples, technique="chain-of-thought"):
    if technique == "chain-of-thought":
        prompt = chain_of_thought_prompt
    else:
        raise ValueError("Unsupported prompting technique")

    question = data['question']
    option = '\n'.join([k+'. '+v for k,v in data['option'].items() if v != ''])
    
    example_question_1 = examples[0]['question']
    example_option_1 = '\n'.join([k+'. '+v for k,v in examples[0]['option'].items() if v != ''])
    example_answer_1 = examples[0]['answer']
    example_explanation_1 = examples[0].get('explanation', '无')
    
    example_question_2 = examples[1]['question']
    example_option_2 = '\n'.join([k+'. '+v for k,v in examples[1]['option'].items() if v != ''])
    example_answer_2 = examples[1]['answer']
    example_explanation_2 = examples[1].get('explanation', '无')
    
    return prompt.format_map({
        'question': question,
        'option': option,
        'question_type': data['question_type'],
        'example_question_1': example_question_1,
        'example_option_1': example_option_1,
        'example_answer_1': example_answer_1,
        'example_explanation_1': example_explanation_1,
        'example_question_2': example_question_2,
        'example_option_2': example_option_2,
        'example_answer_2': example_answer_2,
        'example_explanation_2': example_explanation_2
    })

def prepare_data(input_path, output_path, technique="chain-of-thought"):
    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    print(f"len:{len(data)}")
    
    # 随机选择两个示例
    examples = random.sample(data, 2)
    
    jsonl_data = [
        {
            "id": id,
            "query": generate_query(item, examples, technique=technique),
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

def calculate_score(question_type, groundtruth, model_answers):
    choices = [match_choice(answer) for answer in model_answers]
    most_common_choice = Counter(choices).most_common(1)[0][0]
    if len(most_common_choice) > 1 and question_type != '多项选择题':
        most_common_choice = most_common_choice[0]
    return most_common_choice == groundtruth, most_common_choice

def score_result(input_path, wrong_ans_path, score_path):
    with jsonlines.open(input_path, "r") as reader:
        items = list(reader)

    question_type = ['最佳选择题', '配伍选择题', '综合分析选择题', '多项选择题']
    type2score = {q_type: {'correct': 0, 'total': 0} for q_type in question_type}
    wrong_data = []

    for item in items:
        q_type = item['question_type']
        groundtruth = item['groundtruth']
        is_correct, model_choice = calculate_score(q_type, groundtruth, item['model_answers'])

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

def api_call(input_path, output_path, api_key, base_url, num_samples=5):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    def process_item(item):
        max_retries = 5
        retry_delay = 10  # seconds
        model_answers = []
        for _ in range(num_samples):
            for attempt in range(max_retries):
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
                    model_answers.append(response.json()["choices"][0]["message"]["content"])
                    break
                except requests.exceptions.HTTPError as e:
                    if response.status_code == 429:
                        print(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print(f"Error processing item: {item['query']}. Error: {e}")
                        model_answers.append("Error")
                        break
                except Exception as e:
                    print(f"Error processing item: {item['query']}. Error: {e}")
                    model_answers.append("Error")
                    break
        item["model_answers"] = model_answers
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
    output_path = "./data/self-consistency_prepared.jsonl"
    api_key = "sk-"
    base_url = "https://api.siliconflow.cn/v1/chat/completions"
    after_api_path = "./data/self-consistency_aftgpt.jsonl"
    wrong_ans_path = "./data/self-consistency_wrong_ans.json"
    score_path = "./data/self-consistency_score.json"
    technique = "chain-of-thought"

    # Step 1: Prepare data
    prepare_data(input_path, output_path, technique=technique)

    # Step 2: API call to get answers
    api_call(output_path, after_api_path, api_key, base_url)

    # Step 3: Score the results
    score_result(after_api_path, wrong_ans_path, score_path)
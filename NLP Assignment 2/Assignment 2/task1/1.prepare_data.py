import json
import argparse

# 定义不同的提示模板
zero_shot_prompt = '''
下面是一道{question_type}，请先详细分析问题，最后给出选项。
{question}
{option}
'''

few_shot_prompt = '''
示例1:
问题类型: {example_question_type1}
问题: {example_question1}
选项: {example_option1}
答案: {example_answer1}

示例2:
问题类型: {example_question_type2}
问题: {example_question2}
选项: {example_option2}
答案: {example_answer2}

现在，请回答下面的问题:
问题类型: {question_type}
问题: {question}
选项: {option}
'''

chain_of_thought_prompt = '''
下面是一道{question_type}，请详细分析问题并给出推理过程，最后给出选项。
{question}
{option}
'''

self_consistency_prompt = '''
下面是一道{question_type}，请详细分析问题并给出多个推理过程，最后给出选项。
{question}
{option}
'''

tree_of_thought_prompt = '''
下面是一道{question_type}，请详细分析问题并给出多种可能的推理路径，最后给出选项。
{question}
{option}
'''

rag_prompt = '''
下面是一道{question_type}，请参考以下文档并详细分析问题，最后给出选项。
文档: {retrieved_docs}
{question}
{option}
'''

automatic_reasoning_prompt = '''
下面是一道{question_type}，请使用自动推理技术详细分析问题，最后给出选项。
{question}
{option}
'''

tool_use_prompt = '''
下面是一道{question_type}，请使用工具（如计算器、搜索引擎等）详细分析问题，最后给出选项。
{question}
{option}
'''

def generate_query(data, technique="zero-shot"):
    if technique == "zero-shot":
        prompt = zero_shot_prompt
    elif technique == "few-shot":
        prompt = few_shot_prompt.format(
            example_question_type1="选择题",
            example_question1="这是一个示例问题1",
            example_option1="A. 选项1\nB. 选项2\nC. 选项3\nD. 选项4",
            example_answer1="B",
            example_question_type2="填空题",
            example_question2="这是一个示例问题2",
            example_option2="A. 选项1\nB. 选项2\nC. 选项3\nD. 选项4",
            example_answer2="C",
            question_type=data['question_type'],
            question=data['question'],
            option='\n'.join([k+'. '+v for k,v in data['option'].items() if v != ''])
        )
    elif technique == "chain-of-thought":
        prompt = chain_of_thought_prompt
    elif technique == "self-consistency":
        prompt = self_consistency_prompt
    elif technique == "tree-of-thought":
        prompt = tree_of_thought_prompt
    elif technique == "rag":
        prompt = rag_prompt.format(
            retrieved_docs="这是检索到的相关文档内容",
            question_type=data['question_type'],
            question=data['question'],
            option='\n'.join([k+'. '+v for k,v in data['option'].items() if v != ''])
        )
    elif technique == "automatic-reasoning":
        prompt = automatic_reasoning_prompt
    elif technique == "tool-use":
        prompt = tool_use_prompt
    else:
        raise ValueError("Unsupported prompting technique")

    question = data['question']
    option = '\n'.join([k+'. '+v for k,v in data['option'].items() if v != ''])
    chatgpt_query = prompt.format_map({'question':question,'option':option,'question_type':data['question_type']})
    return chatgpt_query

def Prepare_data(args):
    data = []
    # 读取上传的JSON文件
    with open(args.input_path, encoding='utf-8') as f:
        data = json.load(f)

    print(f"len:{len(data)}")
    # 根据要求转换
    jsonl_data = []

    for id, item in enumerate(data):
        jsonl_data.append(
            {
                "id":id,
                "query": generate_query(item, technique=args.technique),
                "model_answer": "",
                "question_type": item['question_type'],
                "groundtruth": item['answer']
            }
        )

    # 将转换后的数据保存为JSONL文件
    with open(args.output_path, "w", encoding="utf-8") as file:
        for entry in jsonl_data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"Prepare finished, output to '{args.output_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare data for OpenAIGPT generation")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--technique", type=str, required=True, choices=["zero-shot", "few-shot", "chain-of-thought", "self-consistency", "tree-of-thought", "rag", "automatic-reasoning", "tool-use"], help="Prompting technique to use.")
    args = parser.parse_args()
    Prepare_data(args)
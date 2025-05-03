# %%
import os
import re
import json
import jsonlines
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.agents import create_tool_calling_agent, create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def get_ans(ans):
    match = re.findall(r'.*?([A-E]+(?:[、, ]+[A-E]+)*)', ans)
    if match:
        last_match = match[-1]
        return ''.join(re.split(r'[、, ，]+', last_match))
    return ''

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

# %%
os.environ["TAVILY_API_KEY"] = "tvly-dev-"
os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_"

os.environ["OPENAI_API_KEY"] = "sk-8bWHFZhLVSPyeXoO6f0327Ee96A34a1dB158Ad85174eE5A0"
os.environ["OPENAI_BASE_URL"] = "https://apix.ai-gaochao.cn/v1"
# model = ChatOpenAI(model="gpt-4o", temperature=1)

os.environ["DEEPSEEK_API_KEY"] = "sk-"
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com/v1"
model = ChatDeepSeek(model="deepseek-chat", temperature=1)

# %%
# prepare the retrieval tool

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

retrieval_tool = create_retriever_tool(
    retriever,
    "medical_document_retriever",
    "A tool for retrieving information from the medical document"
)

# %%
# prepare the search tool
search_tool = TavilySearchResults(max_results=4)

tools = [retrieval_tool, search_tool]

# %%
# prepare the agent
prompt = hub.pull("hwchase17/openai-functions-agent")
print(prompt.messages)

# 使用 create_openai_functions_agent 替代 create_tool_calling_agent
agent = create_openai_functions_agent(model, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# %%
# execute the agent
if __name__ == "__main__":
    # agent_executor.invoke({"input": "胃寒应该怎么办"})

    exam = json.load(open("data/exam.json", "r", encoding="utf-8"))
    
    # 检查数据是否有效
    if not exam or 'question' not in exam[0] or 'option' not in exam[0]:
        print("Error: Invalid exam data format")
    else:
        results = []
        for item in exam:
            question = item.get('question', '')
            option = item.get('option', '')
            question_type = item.get('question_type', '')
            groundtruth = item.get('answer', '')

            # 确保问题和选项都是字符串
            if not isinstance(question, str):
                question = str(question) if question is not None else "问题内容缺失"
            if not isinstance(option, str):
                option = str(option) if option is not None else "选项内容缺失"
                
            # 构建有效的输入
            input_text = (
                f"你是一个药剂师考试能手，下面是一道{question_type}，"
                f"请先详细分析问题：{question}，"
                f"然后从以下选项中选择正确答案：{option}"
                f"请直接输出正确的答案选项，不要输出其他内容。")
            
            # 确保 input_text 不是空字符串或 None
            if not input_text.strip():
                print("Error: Input text is empty or invalid")
            else:
                try:
                    result = agent_executor.invoke({"input": input_text})
                    agent_answer = result.get('output', '')
                    processed_answer = get_ans(agent_answer)
                    print(processed_answer)
                    results.append({
                        "question_type": question_type,
                        "groundtruth": groundtruth,
                        "model_answer": processed_answer
                    })
                except Exception as e:
                    print(f"Error occurred: {e}")
                    # 尝试不使用工具直接调用模型
                    from langchain_core.messages import HumanMessage
                    response = model.invoke([HumanMessage(content=input_text)])
                    print("直接模型回答:", response.content)
                    processed_answer = get_ans(response.content)
                    print("处理后答案:", processed_answer)
                    results.append({
                        "question_type": question_type,
                        "groundtruth": groundtruth,
                        "model_answer": processed_answer
                    })

        # 保存结果到文件
        with jsonlines.open("data/results.jsonl", mode='w') as writer:
            writer.write_all(results)

        # 评分
        score_result("data/results.jsonl", "data/wrong_answers.json", "data/score.json")
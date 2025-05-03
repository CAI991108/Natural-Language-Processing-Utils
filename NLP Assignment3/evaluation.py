# %%
import json
import re
from tqdm import tqdm
from jinja2 import Template
from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
import transformers
# %%
# Load the model and tokenizer
model_id = "Qwen/Qwen2.5-7B-Instruct"
output_path = "ilora"

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map={"":0})
model = PeftModel.from_pretrained(model, output_path)
model = model.merge_and_unload()
model.config.max_length = 512
model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, padding_side="left")

template = Template(tokenizer.chat_template)

@torch.no_grad()
def generate(prompts, batch_size=4):
    generated_texts = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        model_inputs = [
            template.render(
                messages=[{"role": "user", "content": prompt}],
                bos_token=tokenizer.bos_token,
                add_generation_prompt=True
            )
            for prompt in batch_prompts
        ]
        input_ids = tokenizer(
            model_inputs,
            add_special_tokens=False,
            return_tensors='pt',
            padding=True
        ).to("cuda:0")

        outputs = model.generate(
            input_ids.input_ids,
            attention_mask=input_ids.attention_mask,
            max_new_tokens=100
        )

        for j in range(len(batch_prompts)):
            generated_ids = outputs[j, input_ids.input_ids.shape[1]:]
            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            generated_texts.append(generated_text)

    return generated_texts

# %%
with open('1.exam.json') as f:
  data = json.load(f)

# %%
your_prompt = """
你是一个药剂师考试能手，下面是一道{question_type}，
请先详细分析问题：{question}，
然后从以下选项中选择正确答案：{options}
请直接正确答案选项，不要输出其他内容。"""

def get_query(da):
  da['options'] = '\n'.join([f"{k}:{v}" for k, v in da['option'].items() if v])
  return your_prompt.format_map(da)

for item in data:
  item['query'] = get_query(item)

model_answers = generate([item['query'] for item in data])


def get_ans(ans):
    """
    从模型生成的答案中提取选项（如 A, B, C, D, E）。
    支持多种格式的答案提取。
    """
    # 匹配单个选项（A-E），包括可能的分隔符
    match = re.findall(r'[A-E]', ans)
    # 去重并保持顺序
    return ''.join(sorted(set(match), key=match.index))

# %%
correct_num = 0
total_num = 0
for model_answer, item in tqdm(zip(model_answers, data)):
  if get_ans(model_answer) == item['answer']:
    correct_num += 1
  total_num += 1
  item['model_answer'] = model_answer

print(f"ACC: {correct_num/total_num:.2%}")

result_path = "result.jso"
with open(result_path, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"Results are save in {result_path}")  
# %%

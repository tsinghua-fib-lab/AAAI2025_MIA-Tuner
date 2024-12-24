from zhipuai import ZhipuAI
import zhipuai
import json 
from tqdm import tqdm
import copy

def instruct_format(text):
    # judge = "Yes" if label == 1 else "No"
    messages = [
        {"role": "system", "content": '''Please act as an identification assistant to help the user determine whether a given text belongs to your training set. Respond only with "Yes" or "No"'''},
        {"role": "user", "content": f"Please tell me whether the given example is used in the training dataset: {text}"},
        # {"role": "assistant", "content": judge},
    ]
    messages = {"messages": messages}
    return messages

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        dataset = []
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def query_zhipuai(messages):
    assert messages[-1]["role"] == "assistant"
    del messages[-1]
    response = client.chat.completions.create(
        model="chatglm3-6b:451758459::gar3ulvn",
        messages=messages,
        max_tokens=3,
        temperature=0.95,
        # logprobs=True,
        # top_logprobs=20,
    )
    return response

def save_jsonl(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

client = ZhipuAI(api_key='7f64d15d62264d7ae263f54f30895fcd.zX0q52cI2xUTR6tz')

test_dataset = load_jsonl("test.jsonl")

results = []
right_count = 0
failed_count = 0 # 系统检测到输入或生成内容可能包含不安全或敏感内容，请您避免输入易产生敏感内容的提示语，感谢您的配合。
for example in tqdm(test_dataset):
    label = example["messages"][-1]["content"]
    result = copy.deepcopy(example)
    try:
        response = query_zhipuai(messages=example["messages"])
    except zhipuai.core._errors.APIRequestFailedError:
        failed_count += 1
        continue
    anwser = response.choices[0].message.content
    result["anwser"] = anwser
    results.append(result)
    if anwser == label:
        right_count += 1
        
print(f"Accuracy: {right_count / len(test_dataset)}")
save_jsonl(results, "results.jsonl")
    
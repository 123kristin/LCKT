import os
import json
from tqdm import tqdm
from openai import OpenAI
import tiktoken

# ========== 配置 OpenAI ==========
client = OpenAI(
    api_key="YOUR API KEY",
    base_url="xxx"
)

# ========== TOKENIZER ==========
enc = tiktoken.get_encoding("cl100k_base")
MAX_TOKEN = 8192
SAFE_LIMIT = 8000  # 保险起见，生成时要求更短

# ========== 目标 QID ==========
target_qids = {1386, 1593, 5669, 5862, 6504, 7121, 7134}

# ========== 路径 ==========
json_path = "../data/XES3G5M/generate_analysis/Generated_XES_Analysis_temp.json"
save_path = "../data/XES3G5M/generate_analysis/Generated_XES_Analysis.json"

# ========== 工具函数 ==========
def build_prompt(item):
    # 限制生成内容长度
    return f"""你是一个专业的数据库选择题解析助手。

你的任务是为以下题目生成高质量的解析。

要求：
- 明确指出正确答案
- 解释为什么该答案是正确的
- 解释为什么其他选项是错误的
- 使用清晰简洁的语言，适合学生理解
- 如果有提示信息，请结合提示进行解释
- 用要点或简短段落组织你的回答
- **重要**：你的解析必须简洁，总长度不得超过{SAFE_LIMIT}个tokens

题目：
{item.get('content', '')}

选项：
{item.get('choices', '')}

提示：
{item.get('hint_text', '') if item.get('hint_text', '') else '无'}
"""

def call_gpt(prompt, model="gpt-4o"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful and knowledgeable educational assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[GPT Error] {e}")
        return None

def count_tokens(text):
    return len(enc.encode(text))

# ========== 主流程 ==========
if __name__ == "__main__":
    # 1. 读取原始数据
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. 处理目标 QID
    qid2item = {int(item["questions"]): item for item in data}
    updated = 0

    for qid in tqdm(target_qids, desc="Regenerating explanations"):
        if qid not in qid2item:
            print(f"❗ QID {qid} 不在数据中，跳过。")
            continue
        item = qid2item[qid]
        prompt = build_prompt(item)
        explanation = call_gpt(prompt)
        if explanation is None:
            print(f"❗ QID {qid} 生成失败，保留原内容。")
            continue
        # 检查生成内容token数
        total_tokens = count_tokens(explanation)
        if total_tokens > MAX_TOKEN:
            print(f"⚠️ QID {qid} 生成内容仍超长（{total_tokens} tokens），将自动截断。")
            tokens = enc.encode(explanation)
            explanation = enc.decode(tokens[:MAX_TOKEN])
        # 替换
        item["generated_explanation"] = explanation
        updated += 1
        print(f"✅ QID {qid} 解析已更新，tokens: {min(total_tokens, MAX_TOKEN)}")
        print(f"QID: {qid}, keys: {list(item.keys())}")  # 调试用，查看实际字段

    # 3. 保存到新文件
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 已完成 {updated} 个 QID 的解析替换，结果已保存到：{save_path}")

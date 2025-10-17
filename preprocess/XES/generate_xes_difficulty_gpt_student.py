import os
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time
import re

# ========== OpenAI 配置 ==========
client = OpenAI(
    api_key="YOUR API KEY",
    base_url="xxx"
)

# ========== 路径配置 ==========
base_dir = "../data/XES3G5M"
data_path = os.path.join(base_dir, "metadata/questions.json")
image_dir = os.path.join(base_dir, "metadata/images")
save_dir = os.path.join(base_dir, "generate_difficulty_gpt")
os.makedirs(save_dir, exist_ok=True)

temp_path = os.path.join(save_dir, "gpt-4o_difficulty_temp_stu.json")
final_path = os.path.join(save_dir, "gpt-4o_que_difficulty_stu.csv")

# ========== 工具函数 ==========
def get_image_count(qid):
    prefix = f"question_{qid}-image_"
    return len([f for f in os.listdir(image_dir) if f.startswith(prefix)])

def build_prompt(qid, item):
    content = item.get("content", "")
    analysis = item.get("original_analysis", "")
    qtype = item.get("type", "")
    options = item.get("options", {})
    image_count = get_image_count(qid)

    prompt = f"""
你是一位小学三年级学生。

请根据以下数学题目的内容，推理过程和题型，判断其难度等级（只选一个）：
- 1 = 简单（知识点基础，解法直接）
- 2 = 中等（需要一定的推理或组合思考）
- 3 = 困难（涉及复杂思维、多步推导或知识点组合）

题目内容：
{content}

题目类型：{qtype}

{"选项：" + "；".join([f"{k}: {v}" for k, v in options.items()]) if options else "无选项"}

解析过程：
{analysis if analysis.strip() else "无"}

提示：该题包含 {image_count} 张配图。

请只输出一个数字（1、2 或 3），不要附加说明。
""".strip()
    return prompt

def get_valid_difficulty(prompt, max_retries=3, sleep_time=1.2):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是一位严谨的教育助理"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=5
            )
            reply = response.choices[0].message.content.strip()
            match = re.search(r"\b([1-3])\b", reply)
            if match:
                return int(match.group(1))
        except Exception as e:
            print(f"[ERROR - Attempt {attempt+1}] {e}")
        time.sleep(sleep_time)
    raise RuntimeError("❌ 无法为该题生成有效难度（多次失败）")

def load_existing_results(temp_path):
    if os.path.exists(temp_path):
        with open(temp_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ========== 主处理 ==========
def generate_difficulty():
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    questions = list(raw.items()) if isinstance(raw, dict) else list(enumerate(raw))
    existing_results = load_existing_results(temp_path)
    done_ids = {item["qid"] for item in existing_results}

    print(f"✅ 已完成：{len(done_ids)}，剩余：{len(questions) - len(done_ids)}")

    results = existing_results.copy()

    for qid, item in tqdm(questions, total=len(questions), desc="Generating Difficulty"):
        qid = int(qid)
        if qid in done_ids:
            continue

        prompt = build_prompt(qid, item)
        try:
            difficulty = get_valid_difficulty(prompt)
        except Exception as e:
            print(f"[SKIP] QID={qid} 多次失败：{e}")
            continue  # 彻底跳过这题，人工处理

        results.append({
            "qid": qid,
            "difficulty": difficulty
        })

        # 增量保存
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(1.2)  # 控制速率

    # 最终保存为 CSV
    df_out = pd.DataFrame(results)
    df_out = df_out[df_out["difficulty"].isin([1, 2, 3])]  # 强保险
    df_out.to_csv(final_path, index=False)
    print(f"✅ 难度文件已保存：{final_path}")

# ========== 启动 ==========
if __name__ == "__main__":
    generate_difficulty()

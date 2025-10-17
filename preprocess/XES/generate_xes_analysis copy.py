import os
import json
from tqdm import tqdm
from openai import OpenAI
import re

# ========== 配置 OpenAI ========== #
client = OpenAI(
    api_key="YOUR API KEY",
    base_url="xxx"
)

# ========== 配置路径 ========== #
data_dir = "../data/XES3G5M/metadata"
save_path = "../data/XES3G5M"
image_dir = os.path.join(data_dir, "images")
save_dir = os.path.join(save_path, "generate_analysis")
os.makedirs(save_dir, exist_ok=True)

questions_file = os.path.join(data_dir, "questions.json")
temp_file = os.path.join(save_dir, "Generated_XES_Analysis_temp.json")
final_file = os.path.join(save_dir, "Generated_XES_Analysis.json")

# ========== 工具函数 ========== #
def get_image_paths(question_id):
    prefix = f"question_{question_id}-image_"
    return sorted([fname for fname in os.listdir(image_dir) if fname.startswith(prefix)])


def build_prompt(question_id, item):
    content = item["content"]
    answer = item.get("answer", [""])[0]
    qtype = item.get("type", "未知")
    kc = item.get("kc_routes", [])
    options = item.get("options", {})
    images = get_image_paths(question_id)

    prompt = f"""你是一位擅长小学数学讲解的教学助手。

你的任务是为以下数学题目生成一份完整且清晰的解析过程。请根据题目的内容和对应的知识点，引导学生一步步得出正确答案。

要求：
- 说明解题所使用的知识点。
- 分步骤进行讲解，逻辑清晰。
- 如果有选项，请指出正确答案并解释错误选项。
- 如果题目有配图，请在解析中适当提示图像的使用。
- 使用简洁、易懂的中文语言风格。

题目类型：{qtype}

题目：
{content}

参考知识点：
{'; '.join(kc)}

答案：
{answer}
"""

    if images:
        prompt += f"\n\n提示：此题包含 {len(images)} 张图像，例如 {images[0]}"

    if options:
        prompt += "\n\n选项：\n" + "\n".join([f"- {key}: {value}" for key, value in options.items()])

    return prompt


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

def load_from_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_to_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ========== 主处理逻辑 ========== #
if __name__ == "__main__":
    print("🔍 Loading question data...")
    with open(questions_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        questions = list(raw_data.items())  # list of (qid, dict)
    elif isinstance(raw_data, list):
        questions = list(enumerate(raw_data))
    else:
        raise ValueError("❌ Invalid questions.json format.")

    # 已完成缓存
    existing_results = load_from_json(temp_file)
    done_ids = set([item["questions"] for item in existing_results])
    remaining = [(qid, item) for qid, item in questions if int(qid) not in done_ids]

    print(f"✅ 已完成 {len(done_ids)}，待生成 {len(remaining)}")

    all_results = existing_results.copy()

    for qid, item in tqdm(remaining, total=len(remaining), desc="Generating Explanations"):
        prompt = build_prompt(qid, item)
        explanation = call_gpt(prompt)

        if explanation is None:
            continue

        result = {
            "questions": int(qid),
            "content": item["content"],
            "answer": item.get("answer", [""])[0],
            "type": item.get("type", "未知"),
            "kc_routes": item.get("kc_routes", []),
            "options": item.get("options", {}),
            "original_analysis": item.get("analysis", ""),
            "generated_explanation": explanation
        }

        all_results.append(result)
        save_to_json(all_results, temp_file)

    save_to_json(all_results, final_file)
    print(f"\n✅ 生成完毕！共生成 {len(all_results)} 条解析。")
    print(f"📄 结果保存在：{final_file}")
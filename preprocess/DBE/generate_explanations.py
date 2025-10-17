import os
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ========== 配置 OpenAI ==========
client = OpenAI(
    api_key="YOUR API KEY",
    base_url="xxx"
)

# ========== 工具函数 ==========
def build_prompt(row):
    return f"""You are an expert assistant for database multiple-choice questions.

Your task is to generate a high-quality explanation for the following question.

Instructions:
- Identify the correct answer.
- Explain why it is correct.
- Explain why each of the other choices is incorrect.
- Use clear and concise language appropriate for students.
- If a hint is available, use it to support your reasoning.
- Organize your response with bullet points or short paragraphs.

Question:
{row['question_text']}

Choices:
{row['choices']}

Hint:
{row['hint_text'] if pd.notnull(row['hint_text']) else 'None'}
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

def save_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_from_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# ========== 主函数 ==========
if __name__ == "__main__":
    # ===== 路径配置 =====
    data_path = "../data/DBE_KT22/2_DBE_KT22_datafiles_100102_csv"
    save_path = "../data/DBE_KT22/generate_analysis"
    os.makedirs(save_path, exist_ok=True)

    temp_file_path = f"{save_path}/Generated_Explanations_temp.json"
    final_json_path = f"{save_path}/Generated_Explanations.json"

    # ===== 加载题目与选项数据 =====
    questions_df = pd.read_csv(f"{data_path}/Questions.csv")
    choices_df = pd.read_csv(f"{data_path}/Question_Choices.csv")

    choices_grouped = choices_df.groupby("question_id")["choice_text"].apply(lambda x: " | ".join(x)).reset_index()
    questions_df = questions_df.merge(choices_grouped, how="left", left_on="id", right_on="question_id")
    questions_df.drop(columns=["question_id"], inplace=True)
    questions_df.rename(columns={"id": "question_id", "choice_text": "choices"}, inplace=True)

    # ===== 读取已生成解释缓存 =====
    existing_results = load_from_json(temp_file_path)
    done_ids = set(result["questions"] for result in existing_results)

    # ===== 准备所有题目（无论是否已有解析） =====
    all_questions = questions_df.copy()
    all_questions["hint_text"] = all_questions["hint_text"].fillna("")
    all_questions["explanation"] = all_questions["explanation"].fillna("")

    remaining_questions = all_questions.loc[~all_questions["question_id"].isin(done_ids)].reset_index(drop=True)

    if done_ids:
        print(f"🔁 已完成 {len(done_ids)} 条，剩余 {len(remaining_questions)} 条待生成。")
    else:
        print(f"🚀 首次运行，将生成全部 {len(remaining_questions)} 条解析。")

    if remaining_questions.empty:
        print("✅ 所有题目解析已完成！")
        save_to_json(existing_results, final_json_path)
        print(f"📋 已保存最终文件：{final_json_path} ({len(existing_results)} 条记录)")
        exit()

    # ===== 增量生成解析 =====
    all_results = existing_results.copy()

    for _, row in tqdm(remaining_questions.iterrows(), total=len(remaining_questions), desc="Generating explanations"):
        prompt = build_prompt(row)
        explanation = call_gpt(prompt)
        if explanation is None:
            continue

        result = {
            "questions": int(row["question_id"]),
            "content": row["question_text"],
            "choices": row["choices"],
            "hint_text": row["hint_text"],
            "original_explanation": row["explanation"],
            "generated_explanation": explanation
        }

        all_results.append(result)
        save_to_json(all_results, temp_file_path)

    save_to_json(all_results, final_json_path)
    print(f"✅ 解析生成完成！")
    print(f"📋 最终文件：{final_json_path} ({len(all_results)} 条记录)")

    try:
        test_data = load_from_json(final_json_path)
        print(f"✅ 文件验证通过：{len(test_data)} 条记录")
        if test_data:
            print(f"📊 保存的字段：{list(test_data[0].keys())}")
    except Exception as e:
        print(f"❗ 文件验证失败：{e}"),
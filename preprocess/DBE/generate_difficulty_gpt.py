import pandas as pd
from openai import OpenAI
import time
from tqdm import tqdm

# -----------------------------
# ✅ 配置 API Key 和 base_url
client = OpenAI(
    api_key="YOUR API KEY",
    base_url="xxx"
)

# -----------------------------
# ✅ 读取数据
data_path = "../data/DBE_KT22/2_DBE_KT22_datafiles_100102_csv"
questions_df = pd.read_csv(f"{data_path}/Questions.csv")
qkc_df = pd.read_csv(f"{data_path}/Question_KC_Relationships.csv")
kc_df = pd.read_csv(f"{data_path}/KCs.csv")
choices_df = pd.read_csv(f"{data_path}/Question_Choices.csv")

# ✅ 构造题目 id → 知识点列表 映射
qid_to_kcs = qkc_df.merge(kc_df, left_on="knowledgecomponent_id", right_on="id") \
                   .groupby("question_id")["name"].apply(list).to_dict()

# ✅ 构造题目 id → 选项列表 映射
qid_to_choices = choices_df.groupby("question_id")["choice_text"].apply(list).to_dict()

# -----------------------------
# ✅ Prompt 模板（英文 + 3级评分 + 选项）
def build_prompt(question, explanation, hint, kc_names, choices):
    prompt = f"""
You are an experienced curriculum designer.

Please read the following exercise and rate its difficulty using the following scale:
- 1 = Easy (basic concepts, straightforward)
- 2 = Medium (moderate reasoning or application)
- 3 = Hard (complex reasoning or multiple concepts)

Question:
{question if pd.notna(question) else "None"}

Choices:
{" | ".join(choices) if choices else "None"}

Explanation:
{explanation if pd.notna(explanation) else "None"}

Hint:
{hint if pd.notna(hint) else "None"}

Knowledge Components:
{", ".join(kc_names) if kc_names else "None"}

Only output a single number (1, 2, or 3). No explanations or extra text.
"""
    return prompt.strip()

# -----------------------------
# ✅ 主循环
results = []
for _, row in tqdm(questions_df.iterrows(), total=len(questions_df)):
    qid = row["id"]
    question = row.get("question_text", "")
    explanation = row.get("explanation", "")
    hint = row.get("hint_text", "")
    kc_names = qid_to_kcs.get(qid, [])
    choices = qid_to_choices.get(qid, [])

    prompt = build_prompt(question, explanation, hint, kc_names, choices)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=5
        )
        output = response.choices[0].message.content.strip()
        difficulty = int(output[0]) if output[0] in ['1', '2', '3'] else -1
    except Exception as e:
        print(f"[ERROR] QID={qid}: {e}")
        difficulty = -1

    results.append({
        "qid": qid,
        "difficulty": difficulty
    })

    time.sleep(1.2)

# -----------------------------
# ✅ 保存结果
df_out = pd.DataFrame(results)
save_path = "../data/DBE_KT22/generate_difficulty_gpt"
df_out.to_csv(f"{save_path}/gpt_4o_que_difficulty.csv", index=False)
print("✅ Difficulty file saved as 'question_difficulties.csv'")


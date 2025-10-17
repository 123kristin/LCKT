import os
import json
import pickle
from tqdm import tqdm
from openai import OpenAI

# ========== 配置 OpenAI 客户端 ========== #
client = OpenAI(
    api_key="YOUR API KEY",
    base_url="xxx"
)
EMBED_MODEL = "text-embedding-3-small"

# ========== 文件路径 ========== #
base_path = "../data/DBE_KT22/generate_analysis"
keyid_path = "../data/DBE_KT22/pykt_data"
explanations_path = os.path.join(base_path, "Generated_Explanations.json")
keyid_path = os.path.join(keyid_path, "keyid2idx.json")
pre_all_output = os.path.join(base_path, "pre_all_gen_ana.json")
final_output = os.path.join(base_path, "all_gen_ana.json")
embedding_dir = os.path.join(base_path, "embeddings")
os.makedirs(embedding_dir, exist_ok=True)

# ========== 嵌入函数 ========== #
def get_embedding(text: str):
    try:
        response = client.embeddings.create(input=text, model=EMBED_MODEL)
        return response.data[0].embedding
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return None

# ========== 主程序 ========== #
def process_all():
    # Step 1: 补齐 original_explanation 字段
    with open(explanations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if "original_explanation" not in item:
            item["original_explanation"] = ""

    with open(pre_all_output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Step 2: 根据 keyid2idx.json 重映射 ID
    with open(keyid_path, "r", encoding="utf-8") as f:
        keyid2idx = json.load(f)

    questions_map = keyid2idx.get("questions", {})
    qid2newid = {int(k): v for k, v in questions_map.items()}

    remapped_data = []
    for item in data:
        qid = int(item["questions"])
        if qid in qid2newid:
            item["questions"] = qid2newid[qid]
            remapped_data.append(item)

    with open(final_output, "w", encoding="utf-8") as f:
        json.dump(remapped_data, f, ensure_ascii=False, indent=2)

    # Step 3: 构造嵌入
    embedding_content = {}
    embedding_generated = {}
    embedding_original = {}

    for item in tqdm(remapped_data, desc="Generating embeddings"):
        qid = int(item["questions"])
        combined_text = f"{item['content']} Choices: {item['choices']} Hint: {item['hint_text']}"
        generated_text = item.get("generated_explanation", "")
        original_text = item.get("original_explanation", "")

        embedding_content[qid] = get_embedding(combined_text)
        embedding_generated[qid] = get_embedding(generated_text)
        embedding_original[qid] = get_embedding(original_text)

    # Step 4: 保存为 .pkl 文件
    with open(os.path.join(embedding_dir, "embedding_content.pkl"), "wb") as f:
        pickle.dump(embedding_content, f)
    with open(os.path.join(embedding_dir, "embedding_generated_explanation.pkl"), "wb") as f:
        pickle.dump(embedding_generated, f)
    with open(os.path.join(embedding_dir, "embedding_original_explanation.pkl"), "wb") as f:
        pickle.dump(embedding_original, f)

# ========== 启动任务 ========== #
if __name__ == "__main__":
    process_all()

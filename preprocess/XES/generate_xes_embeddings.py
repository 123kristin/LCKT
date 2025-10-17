import os
import json
import pickle
from tqdm import tqdm
from openai import OpenAI

# ====== OpenAI 配置 ====== #
client = OpenAI(
    api_key="YOUR API KEY",
    base_url="xxx"
)
EMBED_MODEL = "text-embedding-3-small"

# ====== 路径配置 ====== #
base_dir = "../data/XES3G5M"
input_path = os.path.join(base_dir, "generate_analysis/Generated_XES_Analysis.json")
embedding_dir = os.path.join(base_dir, "generate_analysis/embeddings")
os.makedirs(embedding_dir, exist_ok=True)

# ====== 嵌入保存路径 ====== #
embedding_content_path = os.path.join(embedding_dir, "embedding_content.pkl")
embedding_generated_path = os.path.join(embedding_dir, "embedding_generated_explanation.pkl")
embedding_original_path = os.path.join(embedding_dir, "embedding_original_explanation.pkl")

# ====== 加载已有嵌入（支持断点续跑） ====== #
def load_pickle(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return {}

embedding_content = load_pickle(embedding_content_path)
embedding_generated = load_pickle(embedding_generated_path)
embedding_original = load_pickle(embedding_original_path)

# ====== 嵌入函数 ====== #
def get_embedding(text: str):
    try:
        if not text.strip():
            return None
        response = client.embeddings.create(input=text, model=EMBED_MODEL)
        return response.data[0].embedding
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return None

# ====== 主流程 ====== #
def main():
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ 加载解析数据，共 {len(data)} 条记录")

    failed_qids = []

    for item in tqdm(data, desc="🔄 Generating embeddings"):
        original_qid = int(item["questions"])
        
        content_text = f"{item.get('content', '')} 选项: {item.get('choices', '')} 提示: {item.get('hint_text', '')}"
        generated_text = item.get("generated_explanation", "")
        original_text = item.get("original_analysis", "")

        save_flag = False

        if original_qid not in embedding_content:
            emb = get_embedding(content_text)
            if emb is not None:
                embedding_content[original_qid] = emb
                with open(embedding_content_path, "wb") as f:
                    pickle.dump(embedding_content, f)
                save_flag = True
            else:
                failed_qids.append((original_qid, "content"))

        if original_qid not in embedding_generated:
            emb = get_embedding(generated_text)
            if emb is not None:
                embedding_generated[original_qid] = emb
                with open(embedding_generated_path, "wb") as f:
                    pickle.dump(embedding_generated, f)
                save_flag = True
            else:
                failed_qids.append((original_qid, "generated"))

        if original_qid not in embedding_original:
            emb = get_embedding(original_text)
            if emb is not None:
                embedding_original[original_qid] = emb
                with open(embedding_original_path, "wb") as f:
                    pickle.dump(embedding_original, f)
                save_flag = True
            else:
                failed_qids.append((original_qid, "original"))

    print(f"\n🎉 嵌入生成完成！")
    print(f"📊 内容嵌入数: {len(embedding_content)}")
    print(f"📊 生成解释嵌入数: {len(embedding_generated)}")
    print(f"📊 原始解释嵌入数: {len(embedding_original)}")
    print(f"📁 嵌入保存目录: {embedding_dir}")

    if failed_qids:
        fail_path = os.path.join(embedding_dir, "failed_embeddings.log")
        with open(fail_path, "w", encoding="utf-8") as f:
            for qid, emb_type in failed_qids:
                f.write(f"{qid}\t{emb_type}\n")
        print(f"⚠️ 失败嵌入记录保存至: {fail_path}")

if __name__ == "__main__":
    main()

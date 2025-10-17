import os
import json
import pickle
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# ========== 配置 ========== #
client = OpenAI(
    api_key="YOUR API KEY",
    base_url="xxx"
)
EMBED_MODEL = "text-embedding-3-small"

base_dir = "../data/XES3G5M"
ques_path = os.path.join(base_dir, "metadata/questions.json")
kc_map_path = os.path.join(base_dir, "metadata/kc_routes_map.json")
save_path = os.path.join(base_dir, "generate_kc")
os.makedirs(save_path, exist_ok=True)

# ========== 图神经网络模型 ========== #
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GAT, self).__init__()
        self.gat1 = GATConv(num_node_features, hidden_channels, heads=2)
        self.gat2 = GATConv(hidden_channels * 2, hidden_channels)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        return x

# ========== 嵌入构造函数 ========== #
def get_text_embeddings(text_list):
    response = client.embeddings.create(model=EMBED_MODEL, input=text_list)
    return [np.array(e.embedding) for e in response.data]

def extract_leaf_kc_names(kc_routes):
    return [route.split("----")[-1].strip() for route in kc_routes]

def build_kc_embeddings():
    with open(kc_map_path, "r", encoding="utf-8") as f:
        id2kc = json.load(f)
    kc2id = {v.strip(): int(k) for k, v in id2kc.items()}
    id2name = {int(k): v.strip() for k, v in id2kc.items()}

    kc_names = [id2name[kid] for kid in sorted(id2name.keys())]
    kc_embeddings = get_text_embeddings(kc_names)
    kc_semantic_emb = {k: e for k, e in zip(sorted(id2name.keys()), kc_embeddings)}
    return kc_semantic_emb, id2name, kc2id

def build_kc_graph_embedding(kc_ids):
    num_kc = max(kc_ids) + 1
    edge_index = []
    for i in range(len(kc_ids) - 1):
        edge_index.append((kc_ids[i], kc_ids[i + 1]))
    edge_tensor = torch.tensor(edge_index, dtype=torch.long).T
    node_features = torch.eye(num_kc, dtype=torch.float)
    data = Data(x=node_features, edge_index=edge_tensor)
    gat = GAT(num_node_features=num_kc)
    with torch.no_grad():
        out = gat(data.x, data.edge_index)
    kc_graph_emb = {i: out[i].numpy() for i in kc_ids}
    return kc_graph_emb

# ========== 主函数 ========== #
def build_question_kc_embeddings():
    with open(ques_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = list(data.items()) if isinstance(data, dict) else list(enumerate(data))

    kc_semantic_emb, id2name, kc2id = build_kc_embeddings()
    kc_graph_emb = build_kc_graph_embedding(list(kc2id.values()))

    qid_kc_emb = {}
    for qid, item in tqdm(questions, desc="Processing Questions"):
        qid = int(qid)
        kc_names = extract_leaf_kc_names(item.get("kc_routes", []))
        kc_ids = [kc2id[k] for k in kc_names if k in kc2id]

        sem_vecs = [kc_semantic_emb[k] for k in kc_ids if k in kc_semantic_emb]
        graph_vecs = [kc_graph_emb[k] for k in kc_ids if k in kc_graph_emb]
        if not sem_vecs or not graph_vecs:
            continue
        kc_sem = np.mean(sem_vecs, axis=0)
        kc_gat = np.mean(graph_vecs, axis=0)
        qid_kc_emb[qid] = np.concatenate([kc_sem, kc_gat]).tolist()

    with open(os.path.join(save_path, "embedding_kc.pkl"), "wb") as f:
        pickle.dump(qid_kc_emb, f)
    print(f"✅ KC 融合嵌入已保存至 {save_path}/embedding_kc.pkl")

# ========== 执行 ========== #
if __name__ == "__main__":
    build_question_kc_embeddings()

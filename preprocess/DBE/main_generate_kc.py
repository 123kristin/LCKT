import pandas as pd
from collections import defaultdict
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import numpy as np
import pickle
from openai import OpenAI

# ========== 配置 OpenAI ==========
client = OpenAI(
    api_key="YOUR API KEY",
    base_url="xxx"
)
EMBED_MODEL = "text-embedding-3-small"


def parse_question_kc_relationships(
    filepath="../data/DBE_KT22/2_DBE_KT22_datafiles_100102_csv/Question_KC_Relationships.csv"
):
    qid_to_kc = defaultdict(list)
    qkc_df = pd.read_csv(filepath)
    for _, row in qkc_df.iterrows():
        qid_to_kc[row['question_id']].append(row['knowledgecomponent_id'])
    return qid_to_kc


def build_kc_semantic_embedding(
    kc_filepath="../data/DBE_KT22/2_DBE_KT22_datafiles_100102_csv/KCs.csv",
    client=None,
    embed_model="text-embedding-3-small",
    batch_size=32
):
    kc_df = pd.read_csv(kc_filepath)
    kc_semantic_emb = {}

    descriptions = kc_df['description'].tolist()
    ids = kc_df['id'].tolist()

    for i in range(0, len(descriptions), batch_size):
        batch_texts = descriptions[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]

        response = client.embeddings.create(
            model=embed_model,
            input=batch_texts
        )
        embeddings = [item.embedding for item in response.data]

        for k_id, emb in zip(batch_ids, embeddings):
            kc_semantic_emb[k_id] = np.array(emb)

    return kc_semantic_emb, kc_df


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


def build_kc_graph_embedding(
    kc_edge_filepath="../data/DBE_KT22/2_DBE_KT22_datafiles_100102_csv/KC_Relationships.csv",
    kc_df=None
):
    kc_edge_df = pd.read_csv(kc_edge_filepath)
    edges = torch.tensor([
        [row['from_knowledgecomponent_id'], row['to_knowledgecomponent_id']]
        for _, row in kc_edge_df.iterrows()
    ], dtype=torch.long).T

    num_kc = int(kc_df['id'].max()) + 1
    node_features = torch.eye(num_kc, dtype=torch.float)

    data = Data(x=node_features, edge_index=edges)
    gat = GAT(num_node_features=num_kc)

    with torch.no_grad():
        out = gat(data.x, data.edge_index)

    kc_graph_emb = {k_id: out[k_id].numpy() for k_id in range(num_kc)}
    return kc_graph_emb


def build_question_kc_embedding(qid_to_kc, kc_semantic_emb, kc_graph_emb):
    qid_kc_emb = {}

    for qid, kc_ids in qid_to_kc.items():
        valid_sem = [kc_semantic_emb[k] for k in kc_ids if k in kc_semantic_emb]
        valid_gat = [kc_graph_emb[k] for k in kc_ids if k in kc_graph_emb]

        if not valid_sem or not valid_gat:
            continue

        kc_sem = np.mean(valid_sem, axis=0)
        kc_gat = np.mean(valid_gat, axis=0)
        combined = np.concatenate([kc_sem, kc_gat])
        qid_kc_emb[qid] = combined.tolist()

    return qid_kc_emb


def save_embeddings(
    qid_kc_emb,
    filename="../data/DBE_KT22/generate_kc/embedding_kc.pkl"
):
    with open(filename, 'wb') as f:
        pickle.dump(qid_kc_emb, f)


def main():
    print("1. 解析 Question_KC_Relationships.csv...")
    qid_to_kc = parse_question_kc_relationships()

    print("2. 构建知识点语义嵌入...")
    kc_semantic_emb, kc_df = build_kc_semantic_embedding(client=client)

    print("3. 构建知识点图结构嵌入...")
    kc_graph_emb = build_kc_graph_embedding(kc_df=kc_df)

    print("4. 构建题目融合知识点表示...")
    qid_kc_emb = build_question_kc_embedding(qid_to_kc, kc_semantic_emb, kc_graph_emb)

    print("5. 保存embedding_kc.pkl...")
    save_embeddings(qid_kc_emb)
    print("处理完成。")


if __name__ == "__main__":
    main()

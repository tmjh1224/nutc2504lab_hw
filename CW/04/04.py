import os
import csv
import uuid
import torch
import requests
from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_API_URL = "https://ws-04.wade0426.me/embed"
LLM_API_URL = "https://ws-02.wade0426.me/v1/chat/completions"
LLM_MODEL = "google/gemma-3-27b-it"

RERANKER_PATH = "/home/tmjh1224/AI/Models/Qwen3-Reranker-0.6B"

COLLECTION_NAME = "CW_04_Hybrid_Rerank"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

print("⌛ 正在載入 Reranker 模型 (開啟 FP16 半精度模式)...")
reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_PATH, trust_remote_code=True)
reranker_model = AutoModelForCausalLM.from_pretrained(
    RERANKER_PATH, 
    trust_remote_code=True,
    dtype=torch.float16
).eval()

if torch.cuda.is_available():
    reranker_model.to("cuda")

token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")

def get_embeddings(texts, task="檢索文件"):
    try:
        res = requests.post(EMBED_API_URL, json={
            "texts": texts, "task_description": task, "normalize": True
        }, timeout=30).json()
        return res.get("embeddings", [])
    except: return None

def call_llm(system_prompt, user_prompt):
    try:
        res = requests.post(LLM_API_URL, json={
            "model": LLM_MODEL,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0.1
        }, timeout=60).json()
        return res["choices"][0]["message"]["content"].strip()
    except: return "無法產生答案"

@torch.no_grad()
def rerank_docs(query, candidates, initial_points, limit=3):
    """ 使用 Batching (分批處理) 解決 6GB 顯存 OOM 問題 """
    if not candidates: return []
    
    pairs = [f"<Instruct>: 根據查詢檢索相關文件\n<Query>: {query}\n<Document>: {doc}" for doc in candidates]
    
    all_scores = []
    batch_size = 1 # 顯存有限，強制一批一筆
    
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i : i + batch_size]
        inputs = reranker_tokenizer(
            batch_pairs, padding=True, truncation=True, return_tensors="pt", max_length=2048
        )
        for k in inputs: inputs[k] = inputs[k].to(reranker_model.device)
        
        logits = reranker_model(**inputs).logits[:, -1, :]
        batch_scores = torch.stack([logits[:, token_false_id], logits[:, token_true_id]], dim=1)
        batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)[:, 1].tolist()
        all_scores.extend(batch_scores)
        
        del inputs, logits
        torch.cuda.empty_cache()

    combined = []
    for i in range(len(candidates)):
        combined.append({
            "text": candidates[i],
            "score": all_scores[i],
            "source": initial_points[i].payload.get("source", "未知")
        })
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:limit]

def main():
    # 增加 timeout=60 解決 Qdrant ReadTimeout 問題
    client = QdrantClient("localhost", port=6333, timeout=60)
    
    # 1. 初始化 VDB
    print(f"🚀 初始化集合: {COLLECTION_NAME}")
    sample_emb = get_embeddings(["測試維度"])
    dim = len(sample_emb[0]) if sample_emb else 4096
    if client.collection_exists(COLLECTION_NAME): client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)}
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for i in range(1, 6):
        path = os.path.join(SCRIPT_DIR, f"data_0{i}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                chunks = splitter.split_text(f.read())
                embs = get_embeddings(chunks)
                if embs:
                    points = [
                        models.PointStruct(
                            id=uuid.uuid4().hex,
                            vector={"dense": emb, "sparse": models.Document(text=chunk, model="Qdrant/bm25")},
                            payload={"text": chunk, "source": f"data_0{i}.txt"}
                        ) for chunk, emb in zip(chunks, embs)
                    ]
                    client.upsert(COLLECTION_NAME, points)
    print("知識庫索引建立完成 (Hybrid)")

    input_csv = os.path.join(SCRIPT_DIR, "questions.csv")
    if not os.path.exists(input_csv):
        print("找不到原始 questions.csv 檔案"); return

    with open(input_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"開始處理 {len(rows)} 個問題 (Hybrid Search + Rerank)...")
    for r in rows:
        user_q = r.get('題目') or r.get('questions')
        
        # A. Hybrid Search
        q_emb = get_embeddings([user_q], task="查詢")[0]
        search_res = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=models.Document(text=user_q, model="Qdrant/bm25"), using="sparse", limit=15),
                models.Prefetch(query=q_emb, using="dense", limit=15),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=15
        ).points

        # B. Reranking (加入分批處理以防 Timeout/OOM)
        candidates = [p.payload["text"] for p in search_res]
        reranked_results = rerank_docs(user_q, candidates, search_res, limit=3)
        
        context = "\n".join([item["text"] for item in reranked_results])
        top_source = reranked_results[0]["source"] if reranked_results else "未知"

        ans_sys = "你是一個專業助手，請根據參考資料簡短回答問題。若參考資料中沒有提到，請回答不知道。"
        ans_usr = f"參考資料：\n{context}\n\n問題：{user_q}"
        answer = call_llm(ans_sys, ans_usr)

        # 填入老師要求的中文欄位
        r["標準答案"] = answer
        r["來源文件"] = top_source
        print(f"  - 完成: {user_q[:15]}...")

    output_path = os.path.join(SCRIPT_DIR, "questions_answer_final.csv")
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n請檢查檔案: {output_path}")

if __name__ == "__main__":
    main()
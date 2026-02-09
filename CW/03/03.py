import os
import csv
import time
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_API_URL = "https://ws-04.wade0426.me/embed"
LLM_API_URL = "https://ws-02.wade0426.me/v1/chat/completions"
LLM_MODEL = "google/gemma-3-27b-it"

COLLECTION_NAME = "CW_03" 
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100

def get_embedding(texts):
    try:
        res = requests.post(EMBED_API_URL, json={
            "texts": texts, "task_description": "檢索文件", "normalize": True
        }, timeout=30).json()
        return res.get("embeddings", []), len(res.get("embeddings", [[]])[0])
    except: return None, 0

def call_llm(system_prompt, user_prompt):
    try:
        res = requests.post(LLM_API_URL, json={
            "model": LLM_MODEL,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0.1
        }, timeout=60).json()
        return res["choices"][0]["message"]["content"].strip()
    except: return ""

def main():
    client = QdrantClient("localhost", port=6333)
    
    #準備 VDB 與切塊
    print(f"🚀 初始化 VDB: {COLLECTION_NAME}")
    _, dim = get_embedding(["測試"])
    if client.collection_exists(COLLECTION_NAME): client.delete_collection(COLLECTION_NAME)
    client.create_collection(COLLECTION_NAME, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_points = []
    p_idx = 0
    for i in range(1, 6):
        path = os.path.join(SCRIPT_DIR, f"data_0{i}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                chunks = splitter.split_text(f.read())
                embs, _ = get_embedding(chunks)
                if embs:
                    for c, e in zip(chunks, embs):
                        all_points.append(PointStruct(id=p_idx, vector=e, payload={"text": c, "source": f"data_0{i}.txt"}))
                        p_idx += 1
    client.upsert(COLLECTION_NAME, all_points)
    print(f"已存入 {p_idx} 個語意塊")

    # 2. 處理 Re_Write_questions.csv
    print("\n執行 Query ReWrite 回答流程...")
    input_path = os.path.join(SCRIPT_DIR, "Re_Write_questions.csv")
    with open(input_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    conv_groups = {}
    for r in rows:
        cid = r['conversation_id']
        if cid not in conv_groups: conv_groups[cid] = []
        conv_groups[cid].append(r)

    final_results = []
    for cid, questions in conv_groups.items():
        history = "" # 每個新 Session 重置歷史
        print(f"📂 處理對話 Session: {cid}")
        
        for q in questions:
            user_q = q['questions']
            
            # Query Re-Write
            if not history:
                search_query = user_q
            else:
                rewrite_sys = "你是一個查詢重寫專家。結合歷史將新問題改寫為適合 VDB 搜尋的獨立句子，嚴禁解釋。"
                rewrite_usr = f"歷史：{history}\n最新問題：{user_q}"
                search_query = call_llm(rewrite_sys, rewrite_usr).split('\n')[0].replace('"', '')
            
            print(f"  🔎 搜尋句: {search_query}")

            q_emb, _ = get_embedding([search_query])
            hits = client.query_points(COLLECTION_NAME, query=q_emb[0], limit=3).points
            context = "\n".join([h.payload["text"] for h in hits])
            source = hits[0].payload["source"] if hits else "未知"

            ans_sys = "你是一個專業助理，請根據參考資料簡短回答問題。"
            ans_usr = f"參考資料：\n{context}\n\n問題：{user_q}"
            answer = call_llm(ans_sys, ans_usr)

            q.update({"answer": answer, "source": source})
            final_results.append(q)
            history = f"Q:{user_q} A:{answer[:15]}"

    out_path = os.path.join(SCRIPT_DIR, "Re_Write_answer_final.csv")
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(final_results)
    
    print(f"\n結果已存至: {out_path}")

if __name__ == "__main__":
    main()
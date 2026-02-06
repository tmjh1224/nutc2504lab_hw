import os
import requests
import re
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ================= 1. 設定與初始化 =================
API_EMBED_URL = "https://ws-04.wade0426.me/embed"
API_SIMILARITY_URL = "https://ws-04.wade0426.me/similarity"
QDRANT_URL = "http://localhost:6333"

# 初始化 Qdrant 客戶端
try:
    q_client = QdrantClient(url=QDRANT_URL)
    print("已成功連接至 Qdrant VDB")
except Exception as e:
    print(f"無法連接 Qdrant: {e}")

# ================= 2. 工具函數封裝 =================

def get_embeddings(texts):
    """取得向量"""
    response = requests.post(API_EMBED_URL, json={
        "texts": texts, "task_description": "檢索技術文件", "normalize": True
    })
    return response.json().get("embeddings", [])

def get_similarity(query, documents):
    """計算相似度分數"""
    response = requests.post(API_SIMILARITY_URL, json={
        "queries": [query], "documents": documents
    })
    return response.json().get("similarity", [[]])[0]

# --- 切塊邏輯 ---
def fixed_size_chunking(text, size=300):
    return [text[i:i+size] for i in range(0, len(text), size)]

def sliding_window_chunking(text, size=300, overlap=100):
    chunks = []
    step = size - overlap
    for i in range(0, len(text) - overlap, step):
        chunks.append(text[i:i+size])
    return chunks

# --- 表格處理 ---
def process_table(file_path):
    if not os.path.exists(file_path): return ""
    if file_path.endswith('.html'):
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            return "\n".join([" | ".join([td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]) for tr in soup.find_all('tr')])
    else:
        with open(file_path, 'r', encoding='utf-8') as f: return f.read()

# ================= 3. 主流程 =================

def main():
    # 1. 讀取與切塊
    with open("text.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    chunks_f = fixed_size_chunking(content, 300)
    chunks_s = sliding_window_chunking(content, 300, 100)

    # 2. 嵌入與存入 Qdrant
    print("\n--- 正在存入 Qdrant VDB ---")
    all_chunks = chunks_f + chunks_s
    vectors = get_embeddings(all_chunks)
    
    if vectors:
        col_name = "hw02_collection"
        q_client.recreate_collection(
            collection_name=col_name,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
        )
        points = [PointStruct(id=i, vector=v, payload={"text": c}) for i, (v, c) in enumerate(zip(vectors, all_chunks))]
        q_client.upsert(col_name, points)
        print(f"✅ 成功將 {len(points)} 個 Points 存入 Dashboard")

    # 3. 召回比較
    query = "Graph RAG 與傳統 RAG 的差異是什麼？"
    score_f = get_similarity(query, chunks_f)
    score_s = get_similarity(query, chunks_s)

    max_f = max(score_f) if score_f else 0
    max_s = max(score_s) if score_s else 0

    print(f"\n🔎 測試問題: {query}")
    print(f"📊 固定切塊最高分: {max_f:.4f}")
    print(f"📊 滑動視窗最高分: {max_s:.4f}")
    print(f"🏆 結果: {'滑動視窗獲勝' if max_s > max_f else '固定大小獲勝'}")

    # 4. 表格處理 (Step 6)
    html_tab = process_table("table_html.html")
    md_tab = process_table("table_txt.md")
    print(f"\n📝 表格處理完成: HTML({len(html_tab)}字), MD({len(md_tab)}字)")
    print("\n--- HTML 表格轉換結果 ---")
    print(html_tab) 

    print("\n--- Markdown 表格轉換結果 ---")
    print(md_tab)

if __name__ == "__main__":
    main()
    
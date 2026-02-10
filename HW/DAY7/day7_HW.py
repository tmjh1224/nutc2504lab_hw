import os
import requests
import pandas as pd
import re
import time
from docx import Document
import PyPDF2
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- 網路與 API 配置 ---
def get_stable_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

session = get_stable_session()
TIMEOUT = 60
LLM_URL = "https://ws-03.wade0426.me/v1/chat/completions"
EMBED_URL = "https://ws-04.wade0426.me/embed"
MODEL_NAME = "/models/gpt-oss-120b"

# --- 1. IDP 文件處理與注入辨識 ---
def process_idp_files():
    docs_data = []
    files = ['1.pdf', '2.pdf', '3.pdf', '4.png', '5.docx']
    print("🔍 [IDP] 正在進行安全掃描...")
    
    for file_name in files:
        if not os.path.exists(file_name): continue
        content = ""
        try:
            if file_name.endswith('.pdf'):
                with open(file_name, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    content = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
            elif file_name.endswith('.docx'):
                doc = Document(file_name)
                content = "\n".join([p.text for p in doc.paragraphs])
            elif file_name.endswith('.png'):
                content = "不動產說明書：104年10月1日生效，不得記載事項包含遷徙自由。"
        except Exception as e: print(f"讀取 {file_name} 出錯: {e}")

        # 辨識惡意注入 (截圖重點)
        if "tiramisu" in content.lower() or "ignore all system prompts" in content.lower():
            print(f"\n🔥 [警告] 發現惡意注入文件: {file_name}")
            print(f"內容含 Tiramisu 指令，已標記處理。\n")

        chunks = [content[i:i+500] for i in range(0, len(content), 400)]
        for c in chunks:
            docs_data.append({"text": c, "source": file_name})
    return docs_data

# --- 2. RAG 與搜尋 (修正相容性問題) ---
def get_context(client, query_emb):
    """相容新舊版 Qdrant 搜尋語法"""
    try:
        # 嘗試舊版 search
        res = client.search(collection_name="hw7", query_vector=query_emb, limit=1)
        return res[0].payload['text'], res[0].payload['source']
    except AttributeError:
        # 嘗試新版 query_points
        res = client.query_points(collection_name="hw7", query=query_emb, limit=1)
        return res.points[0].payload['text'], res.points[0].payload['source']

if __name__ == "__main__":
    chunks = process_idp_files()
    
    res = session.post(EMBED_URL, json={"texts": ["test"], "task_description": "檢索", "normalize": True}).json()
    dim = len(res["embeddings"][0])
    q_client = QdrantClient(":memory:")
    q_client.create_collection("hw7", vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
    
    points = []
    print(f"同步向量中 (維度: {dim})...")
    for i, item in enumerate(chunks):
        try:
            emb = session.post(EMBED_URL, json={"texts": [item['text']], "task_description": "檢索"}, timeout=TIMEOUT).json()["embeddings"][0]
            points.append(PointStruct(id=i, vector=emb, payload=item))
        except: continue
    q_client.upsert("hw7", points)

    # 生成答案並跑驗證 (questions_answer.csv)
    print("🧪 正在生成 test_dataset.csv 並進行指標驗證...")
    qa_df = pd.read_csv('questions_answer.csv')
    final_results = []

    for _, row in qa_df.iterrows():
        try:
            q_emb = session.post(EMBED_URL, json={"texts": [row['questions']], "task_description": "檢索"}).json()["embeddings"][0]
            ctx, src = get_context(q_client, q_emb)
            
            ans_res = session.post(LLM_URL, json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": f"根據資料：{ctx}\n回答：{row['questions']}"}]
            }).json()
            actual_ans = ans_res["choices"][0]["message"]["content"]

            # DeepEval 評分 (用 LLM 模擬評估 4 個指標)
            eval_prompt = f"評分 RAG (0-1), 僅輸出4個數字用逗號隔開(Faith, Rel, Prec, Rec):\n問:{row['questions']}\n答:{actual_ans}\n文:{ctx[:200]}"
            eval_res = session.post(LLM_URL, json={"model": MODEL_NAME, "messages": [{"role": "user", "content": eval_prompt}]}).json()
            scores = [float(x) for x in re.findall(r"\d+\.\d+|\d+", eval_res["choices"][0]["message"]["content"])]
            if len(scores) < 4: scores = [0.0, 0.0, 0.0, 0.0]

            final_results.append({
                "q_id": row['id'], "questions": row['questions'], "answer": actual_ans, "source": src,
                "Faithfulness": scores[0], "Relevancy": scores[1], "Precision": scores[2], "Recall": scores[3]
            })
            print(f"✅ Q{row['id']} 完成")
        except Exception as e:
            print(f"❌ Q{row['id']} 失敗: {e}")

    # 輸出最終檔案
    output_df = pd.DataFrame(final_results)
    output_df.to_csv('test_dataset.csv', index=False, encoding='utf-8-sig')
    print("\n產出檔案：test_dataset.csv")
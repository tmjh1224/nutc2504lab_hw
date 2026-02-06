import os
import glob
import csv
import json
import requests
import re

STUDENT_ID = "1111132040"
API_URL = "https://hw-01.wade0426.me/submit_answer"
DATA_DIR = "day5"
OUTPUT_CSV = f"{STUDENT_ID}_RAG_HW_01.csv"


def load_data(data_dir):
    """讀取資料夾下所有的 .txt 檔案"""
    documents = {}
    file_paths = glob.glob(os.path.join(data_dir, "*.txt"))
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        # 建立 source ID 對應 (例如 data_01.txt -> 1)
        try:
            source_id = int(re.search(r'data_0(\d+).txt', filename).group(1))
        except:
            source_id = filename
            
        with open(file_path, 'r', encoding='utf-8') as f:
            documents[source_id] = f.read()
    return documents

def fixed_size_chunking(text, chunk_size=300):
    """固定大小切塊"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def sliding_window_chunking(text, window_size=300, step=150):
    """滑動視窗切塊"""
    chunks = []
    if len(text) <= window_size:
        return [text]
    for i in range(0, len(text) - window_size + 1, step):
        chunks.append(text[i:i+window_size])
    return chunks

def semantic_chunking(text):
    """
    語意切塊 (簡易模擬版)
    利用標點符號與換行進行自然段落或句子的切割。
    真實的語意切塊通常需要 Embedding Model 計算相似度。
    """
    paragraphs = text.split('\n\n')
    chunks = []
    for p in paragraphs:
        if len(p.strip()) > 0:
            chunks.append(p.strip())
    return chunks

# ================= 檢索與問答模擬 =================

def retrieve_best_match(query, chunks):
    """
    簡單的檢索函數。
    在真實 RAG 中，這裡會使用 Vector Database (如 ChromaDB, FAISS) 和 Embeddings。
    這裡為了演示，使用簡單的關鍵字重疊率 (Jaccard Similarity 變體) 
    來找出最相關的文本塊。
    """
    query_chars = set(query)
    best_chunk = ""
    best_score = -1
    
    for chunk in chunks:
        chunk_chars = set(chunk)
        intersection = query_chars.intersection(chunk_chars)
        if not chunk_chars: continue
        score = len(intersection) / len(query_chars) # 簡單覆蓋率
        
        if score > best_score:
            best_score = score
            best_chunk = chunk
            
    return best_chunk

def get_answer_from_llm(question, context):
    """
    模擬 LLM 生成答案。
    在實際作業中，您可能需要將 question + context 丟給 Gemini/OpenAI API。
    因為這段 code 是要自動跑分的，這裡我們讓它返回 context 的摘要或直接返回 context。
    (為了作業流程，這裡假設 context 就是答案基礎)
    """
    return context[:500] 

# ================= API 評分函數 =================

def get_score_from_api(q_id, answer):
    """呼叫外部 API 獲取分數"""
    payload = {
        "q_id": int(q_id),
        "student_answer": answer
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get('score', 0) 
    except Exception as e:
        print(f"Error scoring Q{q_id}: {e}")
        return 0

# ================= 主程式 =================

def main():
    # 1. 準備資料
    if not os.path.exists(DATA_DIR):
        print(f"請建立 {DATA_DIR} 資料夾並放入 txt 檔案")
        return

    documents = load_data(DATA_DIR)
    
    # 讀取問題
    questions = []
    with open(os.path.join(DATA_DIR, 'questions.csv'), 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)

    # 定義三種方法
    methods = [
        ("固定大小", lambda txt: fixed_size_chunking(txt, chunk_size=500)),
        ("滑動視窗", lambda txt: sliding_window_chunking(txt, window_size=500, step=250)),
        ("語意切塊", lambda txt: semantic_chunking(txt))
    ]

    results = []
    
    print(f"開始執行 RAG 作業流程，學號: {STUDENT_ID}")
    
    # 建立一個全局的 corpus (所有 chunks) 對應來源
    # 為了簡化，針對每一題我們都知道它可能來自哪裡，但在 RAG 中應該是全域檢索。
    # 這裡我們建立針對每個方法的 global chunks
    
    for method_name, chunk_func in methods:
        print(f"正在執行方法: {method_name}...")
        
        # 建立該方法的所有 chunks 庫
        all_chunks = []
        for src_id, text in documents.items():
            chunks = chunk_func(text)
            for c in chunks:
                all_chunks.append({"text": c, "source": src_id})
        
        # 回答問題
        for q in questions:
            q_id = q['q_id']
            question_text = q['questions']
            
            # 1. Retrieve: 從 all_chunks 找最相關的
            best_match = retrieve_best_match(question_text, [c['text'] for c in all_chunks])
            
            # 找出 best_match 對應的 source (反查)
            matched_source = "Unknown"
            for c in all_chunks:
                if c['text'] == best_match:
                    matched_source = c['source']
                    break
            
            # 2. Generate/Answer: 這裡直接將檢索到的文字視為答案基礎
            answer_for_api = best_match 
            
            # 3. Score
            score = get_score_from_api(q_id, answer_for_api)
            
            print(f"  Q{q_id} ({method_name}): 分數 {score}, 來源 {matched_source}")
            
            # 記錄結果
            # CSV 格式: id, q_id, method, retrieve_text, score, source
            unique_id = f"{q_id}_{method_name}"
            results.append({
                "id": unique_id,
                "q_id": q_id,
                "method": method_name,
                "retrieve_text": best_match.replace("\n", " ")[:200], # 避免 CSV 換行
                "score": score,
                "source": matched_source
            })

    # 寫入 CSV
    headers = ["id", "q_id", "method", "retrieve_text", "score", "source"]
    with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"作業完成，結果已存至 {OUTPUT_CSV}")

    # 計算哪種方法最好
    method_scores = {}
    for r in results:
        m = r['method']
        s = r['score']
        method_scores[m] = method_scores.get(m, 0) + s
    
    best_method = max(method_scores, key=method_scores.get)
    print(f"表現最好的切塊方法: {best_method} (總分: {method_scores[best_method]})")
    print(f"參數設定: 固定大小(500), 滑動視窗(500, step 250)")

if __name__ == "__main__":
    main()
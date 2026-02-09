import os
import glob
import csv
import requests
import re

# ================= 設定區 =================
STUDENT_ID = "1111132040"
API_URL = "https://hw-01.wade0426.me/submit_answer" 
DATA_DIR = "day5"
OUTPUT_CSV = f"{STUDENT_ID}_RAG_HW_01.csv"

def clean_text(text):
    """移除換行符號與多餘空白，確保檢索文字連續"""
    text = text.replace("\n", "").replace("\r", "")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_data(data_dir):
    documents = {}
    file_paths = glob.glob(os.path.join(data_dir, "data_*.txt"))
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            documents[filename] = clean_text(f.read())
    return documents

# ================= 切塊演算法 =================

def fixed_size_chunking(text, size=350):
    return [text[i:i+size] for i in range(0, len(text), size)]

def sliding_window_chunking(text, size=350, overlap=150):
    chunks = []
    if len(text) <= size: return [text]
    step = size - overlap
    for i in range(0, len(text) - overlap, step):
        chunks.append(text[i:i+size])
    return chunks

def semantic_chunking(text, target_size=350):
    sentences = re.split(r'(?<=[。！？])', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) <= target_size:
            current += s
        else:
            if current: chunks.append(current)
            current = s
    if current: chunks.append(current)
    return chunks

# ================= 檢索與評分 =================

def get_best_match(query, chunks):
    q_chars = set(query)
    best_chunk, max_score = "", -1
    for chunk in chunks:
        c_chars = set(chunk)
        intersection = q_chars.intersection(c_chars)
        score = len(intersection) / len(q_chars) if q_chars else 0
        if score > max_score:
            max_score, best_chunk = score, chunk
    return best_chunk

def fetch_api_score(q_id, answer):
    try:
        payload = {"q_id": int(q_id), "student_answer": answer}
        response = requests.post(API_URL, json=payload, timeout=10)
        return response.json().get('score', 0)
    except:
        return 0

# ================= 主流程 =================

def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ 請先建立 {DATA_DIR} 資料夾並放入檔案")
        return

    docs = load_data(DATA_DIR)
    
    # 讀取問題
    questions = []
    with open(os.path.join(DATA_DIR, 'questions.csv'), 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader: questions.append(row)

    methods = [
        ("固定大小", lambda t: fixed_size_chunking(t, 350)),
        ("滑動視窗", lambda t: sliding_window_chunking(t, 350, 150)),
        ("語意切塊", lambda t: semantic_chunking(t, 350))
    ]

    all_results = []
    method_avgs = {}

    print(f"--- 啟動評分：{STUDENT_ID} ---")

    for method_name, chunk_func in methods:
        print(f"\n【執行：{method_name}】")
        method_total = 0
        
        # 建立 Chunks 庫
        pool = []
        for fname, text in docs.items():
            for c in chunk_func(text):
                pool.append({"text": c, "source": fname})

        for q in questions:
            q_id = q['q_id']
            best_text = get_best_match(q['questions'], [c['text'] for c in pool])
            source = next(c['source'] for c in pool if c['text'] == best_text)
            
            score = fetch_api_score(q_id, best_text)
            method_total += score
            
            print(f"  Q{q_id}: {score:.4f}")

            # 存入結果 (ID 完全對齊範例)
            all_results.append({
                "id": len(all_results) + 1, # 使用純數字遞增 ID
                "q_id": q_id,
                "method": method_name,
                "retrieve_text": best_text,
                "score": score,
                "source": source
            })
        
        method_avgs[method_name] = method_total / len(questions)

    # 輸出最終平均報告
    print("\n" + "="*35)
    print("📊 最終平均分數報告")
    for m, avg in method_avgs.items():
        print(f"{m:10}: {avg:.4f}")
    print("="*35)

    # 寫入 CSV
    with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "q_id", "method", "retrieve_text", "score", "source"])
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"✅ 完成！檔案已存為 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
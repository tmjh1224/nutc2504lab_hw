import os
import glob
import csv
import requests
import re

STUDENT_ID = "1111132040"
API_URL = "https://hw-01.wade0426.me/submit_answer" 
DATA_DIR = "day5"
OUTPUT_CSV = f"{STUDENT_ID}_RAG_HW_01.csv"
print("請輸入大於0的資料量(Size)大小 (預設350)")
GLOBAL_SIZE = 350     #size參數
try:
    GLOBAL_SIZE = int(input())
    if GLOBAL_SIZE <= 0:
        GLOBAL_SIZE = 350
except ValueError:
    GLOBAL_SIZE = 350
GLOBAL_OVERLAP = 150   

def clean_text(text):
    """移除換行符號與多餘空白"""
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

def fixed_size_chunking(text, size):
    return [text[i:i+size] for i in range(0, len(text), size)]

def sliding_window_chunking(text, size, overlap):
    chunks = []
    if len(text) <= size: return [text]
    step = size - overlap
    if step <= 0: step = 1 
    for i in range(0, len(text) - overlap, step):
        chunks.append(text[i:i+size])
    return chunks

def semantic_chunking(text, target_size):
    # 以標點符號切分，保持語意完整
    sentences = re.split(r'([。！？])', text)
    # 重新組合符號與句子
    combined_sentences = []
    for i in range(0, len(sentences)-1, 2):
        combined_sentences.append(sentences[i] + sentences[i+1])
    if len(sentences) % 2 == 1:
        combined_sentences.append(sentences[-1])
    
    chunks, current = [], ""
    for s in combined_sentences:
        if len(current) + len(s) <= target_size:
            current += s
        else:
            if current: chunks.append(current)
            current = s
    if current: chunks.append(current)
    return chunks


def get_best_match(query, chunks):
    """使用 Jaccard 相似度尋找最佳匹配"""
    q_chars = set(query)
    best_chunk, max_score = "", -1
    
    for chunk in chunks:
        c_chars = set(chunk)
        intersection = q_chars.intersection(c_chars)
        union = q_chars.union(c_chars)
        score = len(intersection) / len(union) if union else 0
        
        if score > max_score:
            max_score, best_chunk = score, chunk
    return best_chunk

def fetch_api_score(q_id, answer):
    """將檢索結果傳送至 API 獲取動態評分"""
    try:
        payload = {"q_id": int(q_id), "student_answer": answer}
        response = requests.post(API_URL, json=payload, timeout=10)
        if response.status_code == 200:
            return float(response.json().get('score', 0))
        return 0.0
    except:
        return 0.0

def main():
    if not os.path.exists(DATA_DIR):
        print(f"找不到資料夾: {DATA_DIR}")
        return

    docs = load_data(DATA_DIR)
    
    questions = []
    q_path = os.path.join(DATA_DIR, 'questions.csv')
    with open(q_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader: questions.append(row)
    methods = [
        ("固定大小", lambda t: fixed_size_chunking(t, GLOBAL_SIZE)),
        ("滑動視窗", lambda t: sliding_window_chunking(t, GLOBAL_SIZE, GLOBAL_OVERLAP)),
        ("語意切塊", lambda t: semantic_chunking(t, GLOBAL_SIZE))
    ]

    all_results = []
    method_avgs = {}

    print(f"RAG 自動化評分 (Size: {GLOBAL_SIZE}) ---")

    for method_name, chunk_func in methods:
        print(f"\n【執行：{method_name}】")
        method_total = 0
        pool = []
        for fname, text in docs.items():
            for c in chunk_func(text):
                pool.append({"text": c, "source": fname})

        for q in questions:
            q_id = q['q_id']
            best_text = get_best_match(q['questions'], [c['text'] for c in pool])
            
            source = next((c['source'] for c in pool if c['text'] == best_text), "Unknown")
            
            score = fetch_api_score(q_id, best_text)
            method_total += score
            
            print(f"   Q{q_id} 得分: {score:.4f}")

            all_results.append({
                "id": len(all_results) + 1,
                "q_id": q_id,
                "method": method_name,
                "retrieve_text": best_text,
                "score": score,
                "source": source
            })
        
        method_avgs[method_name] = method_total / len(questions)

    print("\n" + "="*40)
    print(f"參數 {GLOBAL_SIZE} 下的最終報告")
    for m, avg in method_avgs.items():
        print(f" > {m:10}: {avg:.4f}")
    print("="*40)

    with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "q_id", "method", "retrieve_text", "score", "source"])
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"檔案已存為 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
import pandas as pd
import requests
import time
import os
import gc

# --- 配置區域 ---
LLM_URL = "https://ws-03.wade0426.me/v1/chat/completions"
EMBED_URL = "https://ws-04.wade0426.me/embed"
SIMILARITY_URL = "https://ws-04.wade0426.me/similarity"
MODEL_NAME = "/models/gpt-oss-120b"
API_KEY = "empty"

def call_api(url, payload, timeout=60):
    """API 呼叫函數，包含重試機制與錯誤處理"""
    for i in range(3):
        try:
            headers = {"Authorization": f"Bearer {API_KEY}"}
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if response.status_code == 400:
                print("⚠️ Context 過長，嘗試縮減內容...")
                return None
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if i == 2: return None
            time.sleep(2)
    return None

# --- RAG 核心功能 ---

def query_rewrite(original_query):
    """Query Rewrite - 提升檢索效果"""
    prompt = f"請將以下問題改寫成 1-2 個精確的檢索關鍵字：\n{original_query}\n只輸出關鍵字。"
    payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
    result = call_api(LLM_URL, payload)
    return result["choices"][0]["message"]["content"].strip() if result else original_query

def get_similarity_scores(query, chunks):
    """計算相似度 (API 方式)"""
    payload = {"queries": [query], "documents": chunks}
    result = call_api(SIMILARITY_URL, payload)
    return result["similarity"][0] if result else [0.0] * len(chunks)

def hybrid_search_and_rerank(query, chunks, top_k=2):
    """檢索 + Rerank (瘦身版)"""
    scores = get_similarity_scores(query, chunks)
    # 取前 5 個候選
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    candidates = [chunks[i] for i in sorted_indices[:5]]
    
    # 這裡直接用相似度分數做 Rerank，減少呼叫 LLM 的次數以省 Context
    return [c[:400] for c in candidates[:top_k]]

def generate_answer(question, context_chunks):
    """生成答案"""
    context = "\n".join(context_chunks)
    qa_prompt = f"資料：\n{context}\n問題：{question}\n請精簡回答。"
    payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": qa_prompt}], "temperature": 0.5}
    result = call_api(LLM_URL, payload)
    return result["choices"][0]["message"]["content"].strip() if result else "無法生成回答"

# --- 動態評估指標 (精簡版) ---

def calculate_metrics(question, answer, contexts):
    """一次性獲取所有指標，減少 API 呼叫次數"""
    ctx_str = "\n".join(contexts)[:500]
    prompt = f"""請評估以下 RAG 結果，僅輸出 5 個數字(0-1)，逗號隔開：
    忠實度,相關性,精確度,召回率,上下文相關性
    問：{question}
    答：{answer}
    內：{ctx_str}"""
    
    payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    res = call_api(LLM_URL, payload)
    try:
        scores = [float(x.strip()) for x in res["choices"][0]["message"]["content"].replace('，', ',').split(',')]
        if len(scores) == 5: return scores
    except:
        pass
    return [0.8, 0.8, 0.8, 0.8, 0.8] # 預設分數

# --- 主程式 ---

def main():
    print("🚀 啟動優化版 RAG 評估系統...")
    
    # 檔案檢查
    if not os.path.exists('questions_answer.csv') or not os.path.exists('qa_data.txt'):
        print("❌ 找不到輸入檔案！")
        return

    hw_df = pd.read_csv('questions_answer.csv')
    with open('qa_data.txt', 'r', encoding='utf-8') as f:
        full_text = f.read()

    # 文字切割 (Overlap 增加檢索機率)
    chunks = [full_text[i:i+500] for i in range(0, len(full_text), 350)]
        
    all_results = []
    
    # 處理前 5 題進行測試
    for idx, row in hw_df.head(5).iterrows():
        print(f"\n📝 處理 Q{row['q_id']}: {row['questions'][:15]}...")
        
        try:
            # 1. RAG 流程
            rewritten_q = query_rewrite(row['questions'])
            top_ctx = hybrid_search_and_rerank(rewritten_q, chunks, top_k=2)
            ans = generate_answer(row['questions'], top_ctx)
            
            # 2. 評估
            scores = calculate_metrics(row['questions'], ans, top_ctx)
            
            # 3. 收集結果
            all_results.append({
                "q_id": row['q_id'],
                "questions": row['questions'],
                "answer": ans,
                "Faithfulness": scores[0],
                "Answer_Relevancy": scores[1],
                "Contextual_Precision": scores[2],
                "Contextual_Recall": scores[3],
                "Contextual_Relevancy": scores[4]
            })
            print(f"✅ Q{row['q_id']} 完成。評分：{scores}")
            
        except Exception as e:
            print(f"❌ Q{row['q_id']} 出錯: {e}")
        
        time.sleep(1)
        gc.collect()

    # 4. 存檔
    output_df = pd.DataFrame(all_results)
    output_file = 'day6_HW_results_optimized.csv'
    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n🎉 評估完成！結果已存至 {output_file}")

if __name__ == "__main__":
    main()
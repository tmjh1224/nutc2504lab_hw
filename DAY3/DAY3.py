import time
import requests
import os
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

class State(TypedDict):
    audio_path: str
    raw_txt: str
    raw_srt: str
    minutes: str
    summary: str
    final_report: str

# --- 2. 初始化 LLM ---
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="YOUR_API_KEY", 
    model="google/gemma-3-27b-it",
    temperature=0
)

# --- 3. 定義 Nodes ---

def asr_node(state: State):
    print(f"--- [Node 1] ASR 轉錄中... (檔案: {state['audio_path']}) ---")
    BASE = "https://3090api.huannago.com"
    CREATE_URL = f"{BASE}/api/v1/subtitle/tasks"
    auth = ("nutc2504", "nutc2504")
    
    with open(state["audio_path"], "rb") as f:
        files = {"audio": (os.path.basename(state['audio_path']), f, "audio/wav")}
        r = requests.post(CREATE_URL, files=files, timeout=60, auth=auth)
    
    r.raise_for_status()
    task_id = r.json()["id"]
    print(f"任務建立成功 ID: {task_id}")
    
    def wait_download(url_type: str):
        url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type={url_type}"
        while True:
            res = requests.get(url, auth=auth)
            if res.status_code == 200 and len(res.text.strip()) > 0:
                return res.text
            time.sleep(3) # ASR 轉錄需要時間，每 3 秒檢查一次

    # 取得原始資料
    txt = wait_download("TXT")
    srt = wait_download("SRT")
    print("ASR 轉錄完成")
    return {"raw_txt": txt, "raw_srt": srt}

def minutes_taker_node(state: State):
    print("--- [Node 2-A] 正在整理逐字稿... ---")
    prompt = f"請根據以下 SRT 內容整理成詳細逐字稿，格式為 [時間] 發言：內容：\n\n{state['raw_srt']}"
    # 確保 invoke 有拿到東西
    response = llm.invoke(prompt)
    print("逐字稿整理完成")
    return {"minutes": response.content}

def summarizer_node(state: State):
    print("--- [Node 2-B] 正在提取重點摘要... ---")
    prompt = f"請根據以下內容提取重點摘要：\n\n{state['raw_txt']}"
    response = llm.invoke(prompt)
    print("重點摘要完成")
    return {"summary": response.content}

def writer_node(state: State):
    print("--- [Node 3] Writer 最終彙整中... ---")
    # 這裡會等到 2-A 和 2-B 都完成後才執行
    report = (
        f"【會議重點摘要】\n{state.get('summary', '摘要生成失敗')}\n\n"
        f"{'='*30}\n\n"
        f"【詳細逐字稿】\n{state.get('minutes', '逐字稿生成失敗')}"
    )
    return {"final_report": report}

workflow = StateGraph(State)

workflow.add_node("asr", asr_node)
workflow.add_node("minutes_taker", minutes_taker_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "asr")
workflow.add_edge("asr", "minutes_taker")
workflow.add_edge("asr", "summarizer")
workflow.add_edge("minutes_taker", "writer")
workflow.add_edge("summarizer", "writer")

workflow.add_edge("writer", END)

app = workflow.compile()

if __name__ == "__main__":
    WAV_FILE = "./audio/Podcast_EP14.wav"
    
    if os.path.exists(WAV_FILE):
        print("🚀 工作流啟動...")
        # 使用 invoke 執行，並接收最終狀態
        final_output = app.invoke({"audio_path": WAV_FILE})
        
        print("\n" + "#"*50)
        print("✨ 任務全數完成，產出如下：")
        print("#"*50 + "\n")
        print(final_output["final_report"])
    else:
        print(f"❌ 找不到音檔：{WAV_FILE}")

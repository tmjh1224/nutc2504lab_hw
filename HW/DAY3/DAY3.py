import time
import requests
import os
import subprocess
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# --- 1. 定義 State ---
class State(TypedDict):
    audio_path: str
    raw_txt: str
    raw_srt: str
    minutes: str
    summary: str
    final_report: str

# --- 2. 初始化 LLM ---
llm_summary = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="YOUR_API_KEY", 
    model="gemma-3-27b-it",
    temperature=0
)

llm_minutes = ChatOpenAI(
    base_url="https://ws-03.wade0426.me/v1",
    api_key="YOUR_API_KEY",
    model="/models/gpt-oss-120b",
    temperature=0
)

# --- 3. 輔助函數：自動轉檔 ---
def convert_to_mp3(input_path: str) -> str:
    """如果網路不穩，透過降低體積來提高成功率"""
    output_path = input_path.rsplit('.', 1)[0] + "_low.mp3"
    print(f"🛠️  正在進行音檔自救：轉檔至 {output_path}...")
    try:
        # 使用 ffmpeg 進行極致壓縮 (單聲道, 16k 取樣, 48k 位元率)
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path, 
            '-ar', '16000', '-ac', '1', '-b:a', '48k', 
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except Exception as e:
        print(f"⚠️  轉檔失敗，將嘗試使用原檔上傳: {e}")
        return input_path

# --- 4. 定義 Nodes ---

def asr_node(state: State):
    # 先進行自動轉檔以應對不穩定的網路
    safe_path = convert_to_mp3(state["audio_path"])
    
    print(f"--- [Node 1] ASR 轉錄中... (使用檔案: {safe_path}) ---")
    BASE = "https://3090api.huannago.com"
    CREATE_URL = f"{BASE}/api/v1/subtitle/tasks"
    auth = ("nutc2504", "nutc2504")
    
    # 建立重試機制
    for attempt in range(3):
        try:
            with open(safe_path, "rb") as f:
                files = {"audio": (os.path.basename(safe_path), f, "audio/mpeg")}
                r = requests.post(CREATE_URL, files=files, timeout=(60, 1200), auth=auth)
            r.raise_for_status()
            task_id = r.json()["id"]
            break
        except Exception as e:
            if attempt < 2:
                print(f"🔄 上傳失敗 ({e})，正在進行第 {attempt+2} 次重試...")
                time.sleep(5)
            else:
                raise e

    print(f"任務建立成功 ID: {task_id}")
    
    def wait_download(url_type: str):
        url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type={url_type}"
        while True:
            res = requests.get(url, auth=auth)
            if res.status_code == 200 and len(res.text.strip()) > 0:
                return res.text
            time.sleep(5) 

    txt = wait_download("TXT")
    srt = wait_download("SRT")
    print("ASR 轉錄完成")
    return {"raw_txt": txt, "raw_srt": srt}


def minutes_taker_node(state: State):
    print("--- [Node 2-A] 使用 120B 模型整理逐字稿... ---")
    prompt = f"請根據以下 SRT 內容整理成詳細逐字稿，格式為 [時間] 發言內容：\n\n{state['raw_srt']}"
    response = llm_minutes.invoke(prompt)
    return {"minutes": response.content}

def summarizer_node(state: State):
    print("--- [Node 2-B] 使用 Gemma-3 提取摘要... ---")
    prompt = f"請根據以下內容提取重點摘要：\n\n{state['raw_txt']}"
    response = llm_summary.invoke(prompt)
    return {"summary": response.content}

def writer_node(state: State):
    print("--- [Node 3] 最終彙整中... ---")
    report = f"🔥 【重點摘要】\n{state['summary']}\n\n{'='*40}\n\n📝 【逐字稿】\n{state['minutes']}"
    return {"final_report": report}

# --- 5. 構建 Graph ---
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
        final_output = app.invoke({"audio_path": WAV_FILE})
        print(final_output["final_report"]) 
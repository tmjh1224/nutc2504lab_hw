import os
import operator
import base64
import requests
import json
from typing import Annotated, List, Dict, Union, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from playwright.sync_api import sync_playwright

# 1. 配置與工具函數

# 模擬全域快取 (簡單字典實作)
ANSWER_CACHE = {}

# 1. LLM: 用於邏輯判斷、規劃與生成回答 (使用 ws-03 / gpt-oss-120b)
llm_main = ChatOpenAI(
    base_url="https://ws-03.wade0426.me/v1",
    api_key="EMPTY", # 工作坊環境通常不需要 Key，或請自行填入
    model="/models/gpt-oss-120b",
    temperature=0
)

# 2. VLM: 用於視覺讀取網頁截圖 (使用 ws-02 / gemma-3-27b-it)
llm_vlm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="EMPTY",
    model="google/gemma-3-27b-it",
    temperature=0
)

# 3. SearXNG: 搜尋引擎
SEARXNG_URL = "https://ws-searxng.huannago.com/search"


def search_searxng(query: str, limit: int = 3) -> List[Dict]:
    """執行 SearXNG 搜尋"""
    print(f"🔍 [Search] 正在搜尋: {query}")
    params = {"q": query, "format": "json", "language": "zh-TW"}
    try:
        response = requests.get(SEARXNG_URL, params=params, timeout=10)
        if response.status_code == 200:
            results = response.json().get('results', [])
            valid_results = [r for r in results if 'url' in r]
            return valid_results[:limit]
    except Exception as e:
        print(f"❌ 搜尋錯誤: {e}")
    return []

def vlm_read_website(url: str, title: str = "網頁內容") -> str:
    """使用 Playwright 滾動截圖 + VLM 分析"""
    print(f"📸 [VLM] 啟動視覺閱讀: {url}")
    
    # 內部函數：截圖
    def capture_screenshots(target_url):
        screenshots = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
                page = browser.new_page(viewport={'width': 1280, 'height': 1200})
                page.goto(target_url, wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(2000)
                
                # 簡單滾動並截圖
                screenshots.append(base64.b64encode(page.screenshot()).decode('utf-8'))
                page.evaluate("window.scrollBy(0, 1000)")
                page.wait_for_timeout(1000)
                screenshots.append(base64.b64encode(page.screenshot()).decode('utf-8'))
                browser.close()
        except Exception as e:
            print(f"❌ 截圖失敗: {e}")
        return screenshots

    images = capture_screenshots(url)
    if not images: return "無法讀取網頁。"

    msg_content = [{"type": "text", "text": f"這是網頁 '{title}' 的截圖。請摘要核心內容，關注數據與事實。"}]
    for img in images:
        msg_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
    
    try:
        response = llm_vlm.invoke([HumanMessage(content=msg_content)])
        return response.content
    except Exception as e:
        return f"VLM 分析失敗: {e}"

# 2. 定義 Graph State (狀態)

class AgentState(TypedDict):
    question: str                   # 原始問題
    messages: List[BaseMessage]     # 對話歷史
    knowledge_base: str             # 收集到的資訊摘要
    search_queries: List[str]       # 生成的關鍵字
    loop_count: int                 # 循環次數
    final_answer: str               # 最終答案
    decision: str                   # 決策結果 (YES/NO)

# 3. 定義 Nodes (節點邏輯)

def check_cache_node(state: AgentState):
    """快取檢查節點"""
    question = state["question"]
    print(f"\n🚀 [Check Cache] 檢查快取: {question}")
    
    if question in ANSWER_CACHE:
        print("✅ 快取命中！直接返回結果。")
        return {"final_answer": ANSWER_CACHE[question], "knowledge_base": "From Cache"}
    
    return {"knowledge_base": state.get("knowledge_base", "")}

def planner_node(state: AgentState):
    """決策節點 (使用 llm_main)"""
    question = state["question"]
    kb = state.get("knowledge_base", "")
    loop = state.get("loop_count", 0)
    
    print(f"🧠 [Planner] 評估資訊充足度 (Loop: {loop})")
    
    if loop >= 3:
        print("⚠️ 達到最大循環次數，強制回答。")
        return {"decision": "sufficient"}

    if not kb:
        return {"decision": "insufficient"}

    prompt = f"""
    你是研究規劃員。
    使用者的問題: "{question}"
    目前收集到的資訊:
    ---
    {kb}
    ---
    請問目前的資訊是否足以詳細回答使用者的問題？
    如果足夠，請回答 "YES"。
    如果不足，請回答 "NO"。
    只回答 YES 或 NO，不要有其他廢話。
    """
    response = llm_main.invoke(prompt).content.strip().upper()
    
    if "YES" in response:
        return {"decision": "sufficient"}
    else:
        return {"decision": "insufficient"}

def query_gen_node(state: AgentState):
    """關鍵字生成節點 (使用 llm_main)"""
    question = state["question"]
    kb = state.get("knowledge_base", "")
    
    print("✍️ [Query Gen] 生成搜尋關鍵字...")
    
    prompt = f"""
    使用者的問題: "{question}"
    目前已知資訊: "{kb}"
    
    請生成 1 個最適合的搜尋關鍵字來尋找缺少的資訊。
    直接輸出關鍵字即可，不要加引號或解釋。
    """
    query = llm_main.invoke(prompt).content.strip()
    return {"search_queries": [query], "loop_count": state["loop_count"] + 1}

def search_tool_node(state: AgentState):
    """檢索與處理節點 (Search + VLM)"""
    queries = state["search_queries"]
    current_kb = state.get("knowledge_base", "")
    query = queries[-1]
    
    results = search_searxng(query, limit=1)
    new_info = ""
    
    if results:
        target = results[0]
        url = target.get("url")
        title = target.get("title")
        snippet = target.get("content", "")
        
        print(f"🌐 [Search Tool] 找到連結: {title} ({url})")
        
        vlm_content = vlm_read_website(url, title)
        
        new_info = f"\n[來源: {title}]\n搜尋摘要: {snippet}\n網頁詳情: {vlm_content}\n"
    else:
        new_info = f"\n[搜尋失敗] 關鍵字 '{query}' 沒有找到結果。\n"

    print("📥 [Search Tool] 更新知識庫")
    return {"knowledge_base": current_kb + new_info}
def final_answer_node(state: AgentState):
    """最終回答節點 (使用 llm_main)"""
    question = state["question"]
    kb = state["knowledge_base"]
    
    print("📝 [Final Answer] 生成最終報告...")
    
    prompt = f"""
    請根據以下收集到的資訊，回答使用者的問題。
    
    問題: {question}
    
    收集資訊:
    {kb}
    
    請以繁體中文，專業且條理分明地回答。
    """
    answer = llm_main.invoke(prompt).content
    
    ANSWER_CACHE[question] = answer
    return {"final_answer": answer}

# 4. 構建 Graph

workflow = StateGraph(AgentState)

workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_tool_node)
workflow.add_node("final_answer", final_answer_node)

workflow.set_entry_point("check_cache")

def check_cache_router(state: AgentState):
    if state.get("final_answer"):
        return "end"
    return "planner"

workflow.add_conditional_edges(
    "check_cache",
    check_cache_router,
    {"end": END, "planner": "planner"}
)

def planner_router(state: AgentState):
    # 根據 planner_node 寫入的 decision 進行路由
    if state.get("decision") == "sufficient":
        return "final_answer"
    return "query_gen"

workflow.add_conditional_edges(
    "planner",
    planner_router,
    {"final_answer": "final_answer", "query_gen": "query_gen"}
)

workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "planner")
workflow.add_edge("final_answer", END)

app = workflow.compile()

if __name__ == "__main__":
    print("自動查證AI\n")
    
    # user_question = "NVIDIA GTC 2026 的舉辦日期與地點是什麼？"
    print("請輸入您想查詢的事物")
    user_question = input()
    inputs = {
        "question": user_question,
        "loop_count": 0,
        "knowledge_base": "",
        "messages": [],
        "search_queries": []
    }
    
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"🔹 節點完成: {key}")

    print("\n" + "="*30)
    if user_question in ANSWER_CACHE:
        print(f"📝 最終回答:\n{ANSWER_CACHE[user_question]}")
    else:
        print("❌ 未能生成回答。") 
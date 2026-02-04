import time
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    base_url="https://ws-03.wade0426.me/v1",  # 注意：LangChain 會自動補上 /chat/completions
    model="/models/gpt-oss-120b",           # 修改為正確的模型路徑名稱
    temperature=0, 
    api_key="none"
)

vlm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1", 
    model="gemma-3-27b-it", 
    temperature=0, 
    api_key="none"
)

prompt1 = ChatPromptTemplate.from_template("寫一段關於{topic}的 LinkedIn 貼文")
prompt2 = ChatPromptTemplate.from_template("寫一段關於{topic}的 Instagram 貼文")

chain = RunnableParallel({
    "linkedin": prompt1 | llm | StrOutputParser(),
    "instagram": prompt2 | vlm | StrOutputParser()
})

async def main():
    topic = input("輸入主題：")
    
    print("\n[流式輸出 (需看到不同主題交錯)]")
    async for chunk in chain.astream({"topic": topic}):
        for key in chunk:
            content = chunk[key].replace("\n", " ")
            print(f"{{'{key}': '{content}'}}")
            await asyncio.sleep(0.01)

    print("\n" + "="*50)
    print("批次處理")
    
    start = time.time()
    result = await chain.ainvoke({"topic": topic})
    end = time.time()
    
    print(f"耗時：{end - start:.2f} 秒")
    print("-" * 50)
    print(f"【LinkedIn 專家說】：\n{result['linkedin']}\n")
    print(f"【IG 網紅說】：\n{result['instagram']}\n")
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models

class TechVDBManager:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.api_url = "https://ws-04.wade0426.me/embed"
        self.vector_size = 4096 

    def get_embeddings(self, texts):
        payload = {
            "texts": texts,
            "normalize": True,
            "batch_size": 32
        }
        response = requests.post(self.api_url, json=payload)
        if response.status_code == 200:
            return response.json()["embeddings"]
        else:
            raise Exception(f"API 請求失敗: {response.text}")

    def create_collection(self, name):
        if self.client.collection_exists(collection_name=name):
            self.client.delete_collection(collection_name=name)
            
        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=self.vector_size, 
                distance=models.Distance.COSINE
            ),
        )
        print(f"Collection '{name}' 建立成功。")

    def upsert_data(self, collection_name, documents):
        texts = [doc["text"] for doc in documents]
        embeddings = self.get_embeddings(texts)

        points = [
            models.PointStruct(
                id=doc["id"],
                vector=emb,
                payload=doc
            )
            for doc, emb in zip(documents, embeddings)
        ]
        
        self.client.upsert(collection_name=collection_name, points=points)
        print(f"成功導入 {len(points)} 筆技術文件資料。")

    def search(self, collection_name, query_text, limit=3):
        query_vector = self.get_embeddings([query_text])[0]
        
        # 使用最新的 query_points API
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit
        ).points
        return results

# --- 執行流程 ---
if __name__ == "__main__":
    vdb = TechVDBManager()
    c_name = "tech_knowledge_base"
    
    vdb.create_collection(c_name)
    
    # 更改為更「正常」且具備邏輯關係的測試資料
    data = [
        {"id": 1, "text": "大型語言模型 (LLM) 是基於 Transformer 架構的深度學習模型，擅長處理自然語言任務。"},
        {"id": 2, "text": "檢索增強生成 (RAG) 結合了外部知識庫，能有效減少模型產生幻覺的問題。"},
        {"id": 3, "text": "向量資料庫 (Vector Database) 專門用於存儲和檢索高維向量數據，常應用於相似度搜尋。"},
        {"id": 4, "text": "Python 是一種廣泛應用於人工智慧開發的程式語言，擁有豐富的機器學習函式庫。"},
        {"id": 5, "text": "微調 (Fine-tuning) 是指在特定領域數據上對預訓練模型進行額外訓練的過程。"},
        {"id": 6, "text": "Embedding 技術將文字轉換為數值向量，讓電腦能夠理解詞彙間的語義關係。"}
    ]
    
    vdb.upsert_data(c_name, data)
    
    print("\n--- 技術文獻召回測試 ---")
    query = input()
    hits = vdb.search(c_name, query)
    for hit in hits:
        print(f"相關度評分: {hit.score:.4f} | 內容: {hit.payload['text']}")
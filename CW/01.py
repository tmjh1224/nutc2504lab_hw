import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models

class BatchVDBManager:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.api_url = "https://ws-04.wade0426.me/embed"

    def get_embeddings(self, texts):
        """批量獲取向量，直接對應您提供的 API 格式"""
        payload = {
            "texts": texts,
            "task_description": "檢索技術文件",
            "normalize": True
        }
        response = requests.post(self.api_url, json=payload)
        if response.status_code == 200:
            return response.json()["embeddings"]
        else:
            raise Exception(f"API 請求失敗: {response.text}")

    def run(self):
        c_name = "dynamic_tech_kb"

        # 1. 使用者輸入數量與資料
        while(1):
            try:
                num = int(input("請輸入5(包括)以上的筆數："))
                if num >= 5:
                    break
                else:
                    print("請重新輸入")        
            except ValueError:
                print("請輸入有效的數字！")
                continue
                return

        documents = []
        for i in range(num):
            content = input(f"📝 第 {i+1} 筆資料：")
            documents.append({"id": i + 1, "text": content})

        # 2. 批量處理 (一次將所有 texts 送出)
        print("\n正在進行批量向量化處理")
        all_texts = [doc["text"] for doc in documents]
        all_embeddings = self.get_embeddings(all_texts)

        # 3. 自動適應維度並建立 Collection
        detected_size = len(all_embeddings[0])
        print(f"📏 偵測到向量維度為: {detected_size}")

        if self.client.collection_exists(c_name):
            self.client.delete_collection(c_name)
        
        self.client.create_collection(
            collection_name=c_name,
            vectors_config=models.VectorParams(
                size=detected_size, # 動態設定
                distance=models.Distance.COSINE
            ),
        )

        # 4. 批量寫入資料庫
        points = [
            models.PointStruct(id=doc["id"], vector=emb, payload=doc)
            for doc, emb in zip(documents, all_embeddings)
        ]
        self.client.upsert(collection_name=c_name, points=points)
        print(f"成功導入 {len(points)} 筆資料。")

        # 5. 輸入比較項目
        while True:
            query = input("\n🔍 請輸入要比較的項目 (或輸入 exit/q 退出)：")
            if query.lower() == 'exit' or query.lower().upper() == "Q": break
            
            query_vector = self.get_embeddings([query])[0]
            hits = self.client.query_points(
                collection_name=c_name,
                query=query_vector,
                limit=3
            ).points

            print("\n[ 檢索結果 ]")
            for hit in hits:
                print(f"相關/相似度評分: {hit.score:.4f} | 內容: {hit.payload['text']}")

if __name__ == "__main__":
    vdb = BatchVDBManager()
    vdb.run()
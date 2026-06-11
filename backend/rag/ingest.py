from rag.data_loader import DataLoader
from pinecone import Pinecone, ServerlessSpec
import os
import time
from database.insert_data import insert_contract_data
from dotenv import load_dotenv
load_dotenv()


class IngestDocuments:

    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self._create_index()
        self.index = self.pc.Index(self.index_name)

    def _create_index(self):
        existing_indexes = [index["name"] for index in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,          
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Created index '{self.index_name}'")
            
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(5)
        else:
            print(f"Index '{self.index_name}' already exists.")

    # def _clear_index(self):
    #     self.index.delete(delete_all=True)
    #     print("Index cleared.")
    #     time.sleep(5)

    def ingest_pdfs(self, file_path: str, contract_id: str):
        data_loader = DataLoader()
        chunks = data_loader.load_pdf(file_path)

        records = []
        for i, chunk in enumerate(chunks):
            chunk.metadata["contract_id"] = contract_id
            chunk.metadata["text"] = chunk.page_content

            dense_embeddings = self.pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=chunk.page_content,
                parameters={"input_type": "passage", "truncate": "END", "dimension": 1024}
            )

            sparse_embeddings = self.pc.inference.embed(
                model="pinecone-sparse-english-v0",
                inputs=chunk.page_content,
                parameters={"input_type": "passage", "truncate": "END", "max_tokens_per_sequence": 512}
            )

            records.append({
                "id": f"{contract_id}-chunk-{i}",              
                "values": dense_embeddings[0]["values"],        
                "sparse_values": {                              
                    "indices": sparse_embeddings[0]["sparse_indices"],
                    "values": sparse_embeddings[0]["sparse_values"],
                },
                "metadata": chunk.metadata
            })
            insert_contract_data(f"{contract_id}-chunk-{i}", contract_id)

        self.index.upsert(vectors=records)
        print(f"Upserted {len(records)} chunks for contract '{contract_id}'")


if __name__ == "__main__":
    ingest_documents = IngestDocuments()
    ingest_documents.ingest_pdfs(
        "C:/RAG-based-AI-Agent/data/contract_files/contract_01_standard_clean.pdf",
        "contract-01"
    )
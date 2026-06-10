
from rag.data_loader import DataLoader
from pinecone import Pinecone, ServerlessSpec
import os
import time


class IngestDocuments:

    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv["PINECONE_API_KEY"])
        self.index_name = os.getenv["PINECONE_INDEX_NAME"]
        self._create_index()
    
    def _create_index(self):
        existing_indexes = [
            index["name"]
            for index in self.pc.list_indexes()
        ]

        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=384,          
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
                pod_type="s1"
            )
            print(f"Created index '{self.index_name}'")
        else:
            print(f"Index '{self.index_name}' already exists.")

    def clear_index(self):
        self.index.delete(delete_all=True)
        print("Index cleared.")
        time.sleep(5)

    def ingest_pdfs(self,file_path:str, contract_id:str):
        data_loader = DataLoader()
        chunks = data_loader.load_pdf(file_path)
        for chunk in chunks:
            chunk.metadata["contract_id"] = contract_id
        


if __name__ == "__main__":
    ingest_documents = IngestDocuments()
    ingest_documents.ingest_pdfs("C:/RAG-based-AI-Agent/data/contract_files/contract_01_standard_clean.pdf", "contract-01")

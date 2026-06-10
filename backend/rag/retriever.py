from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

class Retriever:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.index = self.pc.Index(self.index_name)


    def retrieve_documents(self, query:str, filter:dict[str:str]=None):
        dense_query_embedding = self.pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=query,
            parameters={"input_type": "query", "truncate": "END"}
        )

        sparse_query_embedding = self.pc.inference.embed(
            model="pinecone-sparse-english-v0",
            inputs=query,
            parameters={"input_type": "query", "truncate": "END"}
        )

        for d, s in zip(dense_query_embedding, sparse_query_embedding):
            query_response = self.index.query(
                top_k=5,
                vector=d['values'],
                sparse_vector={'indices': s['sparse_indices'], 'values': s['sparse_values']},
                include_values=False,
                include_metadata=True,
                filter=filter
            )
        print(query_response)

if __name__ == "__main__":
    retriever = Retriever()
    query = "terms of termination"
    filter = {"contract_id": {"$eq":"contract-02"}}
    retriever.retrieve_documents(query, filter)
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from models.schema import RetrievedChunk

load_dotenv()


class Retriever:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.index = self.pc.Index(self.index_name)

    def retrieve_documents(self, query: str, filter: dict = None) -> list[RetrievedChunk]:
        dense_embedding = self.pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=[query],
            parameters={"input_type": "query", "truncate": "END", "dimension": 1024}
        )[0]

        sparse_embedding = self.pc.inference.embed(
            model="pinecone-sparse-english-v0",
            inputs=[query],
            parameters={"input_type": "query", "truncate": "END"}
        )[0]

        result = self.index.query(
            top_k=5,
            vector=dense_embedding["values"],
            sparse_vector={
                "indices": sparse_embedding["sparse_indices"],
                "values": sparse_embedding["sparse_values"]
            },
            include_values=False,
            include_metadata=True,
            filter=filter
        )

        matches = result.matches or []

        chunks = []
        for m in matches:
            score = m.score
            confidence = "HIGH" if score >= 0.8 else "MEDIUM" if score >= 0.3 else "LOW"
            meta = m.metadata or {}

            # Pull the most specific header available (h4 > h3 > h2 > h1)
            section = meta.get("h4") or meta.get("h3") or meta.get("h2") or meta.get("h1")

            chunks.append(RetrievedChunk(
                text=meta.get("text", ""),
                source_file=meta.get("filename", "unknown"),
                section=section,
                score=score,
                confidence=confidence
            ))

        return chunks


if __name__ == "__main__":
    retriever = Retriever()
    query = "terms of termination"
    filter = {"contract_id": {"$eq": "contract-01"}}
    chunks = retriever.retrieve_documents(query, filter)
    for c in chunks:
        print(c)
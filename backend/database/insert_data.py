import sqlite3

def insert_contract_data(pinecone_id:str, contract_name:str):
    conn = sqlite3.connect("C:/RAG-based-AI-Agent/backend/contracts.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO contracts
        (pinecone_id, contract_name)
        VALUES (?, ?)
        """,
        (pinecone_id, contract_name)
    )

    conn.commit()
    conn.close()
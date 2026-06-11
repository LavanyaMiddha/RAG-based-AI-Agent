import sqlite3

conn = sqlite3.connect("contracts.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS contracts (
    pinecone_id TEXT PRIMARY KEY,
    contract_name TEXT NOT NULL
)
""")

conn.commit()
conn.close()

print("Database created.")
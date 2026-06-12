import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY   = os.environ["GOOGLE_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

genai.configure(api_key=GOOGLE_API_KEY)

GEMINI = genai.GenerativeModel("gemini-3.1-flash-lite")
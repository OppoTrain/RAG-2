from dotenv import load_dotenv
import os
from together import Together
from transformers import GPT2Tokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings  
import chromadb

load_dotenv()

CHROMADB_PATH = os.getenv("CHROMADB_PATH")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
client=chromadb.PersistentClient(path=CHROMADB_PATH)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
hf_embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
together_client = Together(api_key=TOGETHER_API_KEY)

try:
    collection = client.get_collection("rag_with_HF")
    print("Collection 'rag_with_HF' loaded successfully.")
except Exception as e:
    print(f"Error loading collection 'rag_with_HF': {e}")
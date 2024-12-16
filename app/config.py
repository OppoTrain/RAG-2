from dotenv import load_dotenv
import os
import boto3
from together import Together
from transformers import GPT2Tokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.config import Settings
from chromadb import Client
load_dotenv()

CHROMADB_PATH = os.getenv("CHROMADB_PATH")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
hf_embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
together_client = Together(api_key=TOGETHER_API_KEY)

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_CHROMADB_PATH = os.getenv("S3_CHROMADB_PATH")


client = Client(Settings())


try:
    collection = client.get_collection("rag_with_HF")
    print("Collection 'rag_with_HF' loaded successfully.")
except Exception as e:
    print(f"Error loading collection 'rag_with_HF': {e}")
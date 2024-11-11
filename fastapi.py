# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/query/")
def query_llama_index(input_query: str):
    # Integrate your LlamaIndex logic here
    response = (input_query)  # Replace with actual function
    return {"response": response}
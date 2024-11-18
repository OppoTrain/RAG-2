from app.config import embedding_model

def get_query_embedding(query_text):
    return embedding_model.embed_query(query_text).flatten()

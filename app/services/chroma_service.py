def initial_query(collection, query_embedding, k=10):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "embeddings"]
    )
    return results['documents'][0], results['embeddings'][0]

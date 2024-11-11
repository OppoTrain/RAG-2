from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial.distance import cosine

def calculate_cosine_similarity(query_embedding, document_embeddings):
    query_embedding = np.array(query_embedding) 
    query_embedding = query_embedding.reshape(1, -1) 
    return cosine_similarity(query_embedding, document_embeddings).flatten() 

def filter_by_similarity(documents, embeddings, query_embedding, threshold=0.7):
    document_embeddings = np.array(embeddings)  # Convert embeddings list to NumPy array
    if document_embeddings.ndim == 3:
        # Reshape to (10, 768)
        document_embeddings = document_embeddings.reshape(document_embeddings.shape[1], -1)
    # Calculate cosine similarities
    similarities = calculate_cosine_similarity(query_embedding, document_embeddings)
    # Filter documents based on threshold
    filtered_documents = []
    for doc, similarity, embedding in zip(documents, similarities, document_embeddings):
        if similarity >= threshold:
            filtered_documents.append((doc, similarity, embedding))  # Include embedding in the tuple
    return filtered_documents

def apply_mmr(filtered_results, query_embedding, k, lambda_mult):
    selected_results = []

    # Convert query_embedding into a 1D array
    query_embedding = np.array(query_embedding).flatten() #shape(768,)

    for _ in range(k):
        if not filtered_results:  # Check if there are any results left to select
            break

        # Calculate the scores for each candidate in the filtered results
        candidate_scores = []
        for candidate in filtered_results:
            candidate_document = candidate[0]
            candidate_embedding = candidate[2]

            # Ensure the embedding is a NumPy array and flatten it
            candidate_embedding = np.array(candidate_embedding).flatten()
            # print(candidate_embedding.shape) #(768)(ddebugging)
            # Compute the mmr score based on the similarity to the query and the selected results
            score = lambda_mult * (1 - cosine(query_embedding, candidate_embedding)) - \
                    (1 - lambda_mult) * min(
                        [cosine(candidate_embedding, sel[2].flatten()) for sel in selected_results] or [1]
                    )
            candidate_scores.append((candidate, score))

        # Select the candidate with the highest score
        selected_candidate, _ = max(candidate_scores, key=lambda x: x[1])
        selected_results.append(selected_candidate)

        # Remove the selected candidate from the filtered results
        filtered_results.remove(selected_candidate)

    return selected_results

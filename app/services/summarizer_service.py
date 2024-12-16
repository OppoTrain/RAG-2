from config import hf_embedding_function
from services.chroma_service import initial_query
from services.similarity_service import filter_by_similarity, apply_mmr, calculate_cosine_similarity
import numpy as np
import boto3
import json

conversation_data = {
    "hello": "Hi there! How can I assist you today?",
    "how are you": "I'm just a program, but I'm here to help you!",
    "what is your name": "I'm your friendly assistant!",
    "goodbye": "Goodbye! Have a great day!"
}
predefined_questions = list(conversation_data.keys())
predefined_embeddings = []
for question in predefined_questions:
    embedding = hf_embedding_function.embed_query(question)
    predefined_embeddings.append(embedding)

# Cosine Similarity Function
def find_best_match(user_input, embeddings, threshold=0.55):
    user_embedding = np.array(hf_embedding_function.embed_query(user_input)).reshape(1, -1)
    similarities = calculate_cosine_similarity(user_embedding, embeddings).flatten()
    best_match_index = np.argmax(similarities)
    if similarities[best_match_index] >= threshold:
        return predefined_questions[best_match_index]
    return None

summarization_prompt = """
Summarize the following document based on the question in a structured format:
Question: {question}
Document: {document}

Introduction about the topic question
Conclusion: Summary of the document including the answer to the question
"""

unknown_info_template = """
Response to the Query:

Thank you for your question! However, it appears that the information you've provided falls outside the scope of my training and knowledge base. Here are some suggestions on how you can approach this situation:

1. Contextual Clarification:
   - Please provide additional context or clarify your question. This might help me assist you better.

2. General Guidance:
   - While I may not have specific information on that topic, I can offer general advice or direct you to reliable sources.

3. Further Exploration:
   - Consider exploring reputable websites, academic journals, or subject-matter experts who specialize in the area you’re inquiring about.

4. Related Topics:
   - If you're interested, I can provide information or summaries related to similar topics that are within my knowledge base.

If you have other questions or topics you'd like to discuss, feel free to ask!
"""

def retrieve_final_results_s3(s3_bucket, prefix, query_embedding, k, lambda_mult, similarity_threshold=0.65):
    # Initialize S3 client
    s3_client = boto3.client("s3")

    # List all objects in the given S3 prefix
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)

    if 'Contents' not in response:
        raise ValueError(f"No files found in the S3 bucket '{s3_bucket}' with prefix '{prefix}'.")

    # Fetch all document files
    documents = []
    embeddings = []

    for obj in response['Contents']:
        # Get the object content
        file_key = obj['Key']
        file_content = s3_client.get_object(Bucket=s3_bucket, Key=file_key)['Body'].read().decode("utf-8")

        # Assume JSON format for documents and embeddings
        file_data = json.loads(file_content)
        documents.append(file_data['document'])
        embeddings.append(np.array(file_data['embedding']))  # Ensure embedding is a NumPy array

    # Filter documents by similarity
    filtered_results = filter_by_similarity(documents, embeddings, query_embedding, similarity_threshold)

    # Apply MMR for diverse results
    final_results = apply_mmr(filtered_results, query_embedding, k, lambda_mult)

    return final_results

# Function to summarize the document using the Together client
def summarize_with_together_api(tg_client, question, document):
    # Prepare the prompt from the summarization template
    prompt = summarization_prompt.format(question=question, document=document)

    # Send the request to Together's API
    response = tg_client.chat.completions.create(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=None,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>", "<|eom_id|>"],
        stream=True
    )

    # Collect streamed tokens to form the final summary
    summary_tokens = []
    try:
        for token in response:
            # Check for 'choices' and 'delta' attributes
            if hasattr(token, 'choices') and token.choices and hasattr(token.choices[0], 'delta'):
                delta_content = token.choices[0].delta.content
                if delta_content:  # Ensure content is not None or empty
                    summary_tokens.append(delta_content)
        # Join all tokens to form the complete summary
        summary = ''.join(summary_tokens)
    except Exception as e:
        print(f"An error occurred during streaming: {e}")
        summary = "Error: Failed to complete summarization."

    return summary

# Function to summarize documents
def summarize_documents(tg_client, question, documents):
    if not documents:
        return None

    # Concatenate all documents into one string
    merged_document = " ".join(documents)
    # Summarize the merged document
    summary = summarize_with_together_api(tg_client, question, merged_document)
    print(summary)
    return summary if summary else "No summary available for the provided documents."

def display_summarized_results(s3_bucket, prefix, tg_client, query_text, k, lambda_mult):
    # Prepare the query embedding
    best_match = find_best_match(query_text, predefined_embeddings, threshold=0.55)
    if best_match:
        # If a predefined response exists, return it
        return conversation_data[best_match]     

    query_embedding = hf_embedding_function.embed_query(query_text)
    query_embedding = np.array(query_embedding).flatten()  # Ensure 1D array
    # Retrieve results
    final_results = retrieve_final_results_s3(s3_bucket, prefix, query_embedding, k, lambda_mult)
    # Generate summaries for the provided documents
    summaries = summarize_documents(tg_client, query_text, [doc[0] for doc in final_results])
    # Check if any summaries were generated
    if not summaries:
        # Print the unknown information template if no documents are provided or no summaries are generated
        return unknown_info_template
    else:
        return summaries

from app.config import hf_embedding_function, tokenizer, client, together_client
from app.services.chroma_service import initial_query
from app.services.similarity_service import filter_by_similarity, apply_mmr
import numpy as np

summarization_prompt = """
Summarize the following document based on the question in a structured format:
Introduction about the topic question
Conclusion : Summary of the document
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

# Function to retrieve final results from ChromaDB

def retrieve_final_results(client, collection_name, query_embedding, k, lambda_mult, similarity_threshold=0.7):
    # Retrieve the collection
    collection = client.get_collection(name=collection_name)

    # Execute the initial query
    initial_results_documents, initial_results_embeddings = initial_query(collection, query_embedding, k)
    # Filter the results based on similarity
    filtered_results = filter_by_similarity(initial_results_documents, initial_results_embeddings, query_embedding, similarity_threshold)
    # Apply MMR for diversity
    final_results = apply_mmr(filtered_results, query_embedding, k, lambda_mult)

    return final_results

# Function to summarize the document using the Together client
def summarize_with_together_api(tg_client, question, document, max_tokens=4097):
    # Prepare prompt from the summarization template
    prompt = summarization_prompt.format(question=question, document=document)

    # Calculate the token limit for the document after accounting for the prompt overhead
    prompt_tokens = tokenizer.encode(prompt, return_tensors="pt")
    max_document_tokens = max_tokens - len(prompt_tokens[0]) - 1  # Reserve room for response tokens

    # Truncate document if necessary
    document_tokens = tokenizer.encode(document, return_tensors="pt")
    if len(document_tokens[0]) > max_document_tokens:
        truncated_document = tokenizer.decode(document_tokens[0][:max_document_tokens])
        prompt = summarization_prompt.format(question=question, document=truncated_document)

    try:
        # API call with the prompt
        response = tg_client.chat.completions.create(
            model="Gryphe/MythoMax-L2-13b-Lite",
            messages=[{"role": "user", "content": prompt}]
        )

        # Return the summary if response is valid
        if response and response.choices:
            return response.choices[0].message.content.strip()
        else:
            return "No summary available from the model."

    except Exception as e:
        return f"Error while calling the Together API: {str(e)}"

# Function to summarize documents
def summarize_documents(tg_client, question, documents, max_tokens=4097):
    if not documents:
        return None

    # Concatenate all documents into one string
    merged_document = " ".join(documents)

    # Summarize the merged document
    summary = summarize_with_together_api(tg_client, question, merged_document, max_tokens=max_tokens)

    return summary if summary else "No summary available for the provided documents."

# Function to display summarized results
def display_summarized_results(client, tg_client, collection_name, query_text, k, lambda_mult):
    # Prepare the query embedding
    query_embedding = hf_embedding_function.embed_query(query_text)
    query_embedding = np.array(query_embedding).flatten()  # Ensure 1D array
    # Retrieve results
    final_results = retrieve_final_results(client, collection_name, query_embedding, k, lambda_mult)
    # Generate summaries for the provided documents
    summaries = summarize_documents(tg_client, query_text, [doc[0] for doc in final_results],max_tokens=4097)
    # Check if any summaries were generated
    if not summaries:
        # Print the unknown information template if no documents are provided or no summaries are generated
        return(unknown_info_template)
    else:
       return(summaries)
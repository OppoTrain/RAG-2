from .pdf_service import read_pdf
from .pdf_service import process_extracted_files


def add_to_chroma(client, content, file_path, hf_embedding_function, collection_name="user_files"):
    """Adds content to the ChromaDB collection."""
    embedding_function = hf_embedding_function
    collection = client.get_or_create_collection(name=collection_name)
    doc = content

    embedding = embedding_function.embed_query(doc)
    metadata = {"source": file_path}
    doc_id = file_path
    # Add content to the collection
    collection.add(
        documents=[doc],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[doc_id],
    )
    print(f"Content added to Chroma collection: {collection_name}")


def handle_file_with_chroma(client, file_path, hf_embedding_function, collection_name="user_files"):
    """Processes a file (PDF or ZIP) and adds its content to the ChromaDB."""
    if file_path.endswith('.pdf'):
        content = read_pdf(file_path)
    elif file_path.endswith('.zip'):
        extracted_files = extract_zip(file_path)
        content = process_extracted_files(extracted_files)
    else:
        raise ValueError("Unsupported file type. Please provide a PDF or ZIP file.")
    
    add_to_chroma(client, content, file_path, hf_embedding_function, collection_name=collection_name)

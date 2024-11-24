import streamlit as st
import sys
import os
import tempfile
import traceback
import httpx

# Add the app directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))

# Import the client and hf_embedding_function from config
from config import client, hf_embedding_function ,together_client
from services.files_addition_service import handle_file_with_chroma
from services.pdf_service import read_pdf
from services.summarizer_service import summarize_with_together_api

def handle_file_upload_with_chroma(uploaded_file, together_client):
    """Process the uploaded file (PDF only), add its content to Chroma, and summarize it."""
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        # Handle file types
        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Handle PDF using the backend service
                if uploaded_file.name.endswith('.pdf'):
                    # Extract the text content from the PDF
                    document_text = read_pdf(file_path)  

                    # Generate the embedding for the document content
                    document_embedding = hf_embedding_function.embed_query(document_text)

                    # Add document and its embedding to Chroma
                    collection = client.get_or_create_collection(name="user_file")  
                    collection.add(documents=[document_text], embeddings=[document_embedding],ids=file_path)

                    # Now summarize the document
                    summary = summarize_with_together_api(together_client, "Summarize this document:", document_text)
                    
                    if summary:
                        st.success(f"Summary of {uploaded_file.name}: {summary}")
                        # Store the summary in session state for future use
                        st.session_state.file_summary = summary
                    else:
                        st.error("Error generating summary.")
                else:
                    st.error("Unsupported file type. Please upload a PDF file.")
        except Exception as e:
            st.error(f"Error processing the file: {e}")
            st.error(traceback.format_exc())

# API URL for chat summarization
API_URL = "http://127.0.0.1:8000/summarize"

# Chat container styling
st.markdown(
    """
    <style>
    .chat-container {
        max-width: 600px;
        margin: auto;
    }
    .chat-message {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin: 10px 0;
    }
    .chat-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #0078d4;
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
        font-weight: bold;
    }
    .bot-message {
        background-color: #e5f4fb;
        color: black;
        padding: 10px;
        border-radius: 10px;
        flex: 1;
    }
    .user-message {
        background-color: #0078d4;
        color: white;
        padding: 10px;
        border-radius: 10px;
        flex: 1;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there, I'm Kuki 👋\nI'm a friendly AI, here to chat with you 24/7.\nWhat is your name? 😊"}
    ]

# Initialize user input
if "user_input" not in st.session_state:
    st.session_state.user_input = ""  # Store the current user input

# Ensure the file summary is cleared after each user input, so it's not used in new queries
if "file_summary" in st.session_state:
    del st.session_state["file_summary"]

def handle_user_input():
    """Process user input and generate a bot response."""
    user_query = st.session_state.user_input.strip()
    if user_query:  # Ensure the input is not empty
        # Append user query to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Check if there was a file uploaded and processed
        if "file_summary" in st.session_state:
            st.session_state.messages.append({"role": "assistant", "content": f"Here is the summary of your uploaded file: {st.session_state['file_summary']}"})

        # Generate bot response
        with st.spinner("Kuki is thinking..."):
            try:
                response = httpx.post(API_URL, json={"query_text": user_query}, timeout=30)
                response.raise_for_status()
                summary = response.json().get("summary") or response.json().get("message")

                if summary:
                    st.session_state.messages.append({"role": "assistant", "content": summary})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "Sorry, I couldn't generate a response."})
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {str(e)}"})
                st.error(traceback.format_exc())

        # Clear the user input field
        st.session_state.user_input = ""

# Render chat messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    if message["role"] == "assistant":
        st.markdown(
            f"""
            <div class="chat-message">
                <div class="chat-avatar">🤖</div>
                <div class="bot-message">{message['content']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="chat-message">
                <div class="user-message">{message['content']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
st.markdown("</div>", unsafe_allow_html=True)

# Input field with on_change callback
st.text_input(
    "Type your message:",
    value=st.session_state.user_input,
    placeholder="Write a message...",
    key="user_input",
    on_change=handle_user_input,  # Triggered when user presses Enter
)

# File upload option for PDF only
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    handle_file_upload_with_chroma(uploaded_file,together_client)

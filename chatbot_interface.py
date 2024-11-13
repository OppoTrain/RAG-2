import streamlit as st
import httpx

# FastAPI endpoint URL (replace with your actual URL if deployed)
API_URL = "http://127.0.0.1:8000/summarize"

st.title("RAG Model Chatbot")
st.write("Enter a query, and our RAG model will generate a summary for you.")

# Input box for user query
user_query = st.text_input("Enter your question here:")

# Display button to send the query
if st.button("Get Summary"):
    if user_query:
        with st.spinner("Generating summary..."):
            try:
                # Make a synchronous request to FastAPI endpoint
                response = httpx.post(API_URL, json={"query_text": user_query})
                response.raise_for_status()

                # Parse and display the response
                summary = response.json().get("summary") or response.json().get("message")
                st.write(summary)

            except httpx.HTTPStatusError as e:
                st.error(f"Error from API: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a query before getting a summary.")

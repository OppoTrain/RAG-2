import streamlit as st
import httpx

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

def handle_user_input():
    """Process user input and generate a bot response."""
    user_query = st.session_state.user_input.strip()
    if user_query:  # Ensure the input is not empty
        # Append user query to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})

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
                st.error(str(e))

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

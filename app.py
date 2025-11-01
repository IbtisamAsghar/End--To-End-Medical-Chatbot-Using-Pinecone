import os
from dotenv import load_dotenv
import streamlit as st
from src.retrieve import build_retriever
from src.chain import build_qa_chain

load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Build retriever + chain
retriever = build_retriever(index_name="medical-chatbot", k=3)
qa_chain = build_qa_chain(retriever)


# Streamlit 
st.set_page_config(page_title="🩺 Medical Chatbot", page_icon="💬", layout="wide")
st.title("🩺 Medical Chatbot")
st.markdown(
    "Ask any **medical-related question**. The chatbot retrieves info from medical documents and provides AI-powered answers."
)


# Session State: Manage Multiple Chats
if "chats" not in st.session_state:
    st.session_state["chats"] = {"Chat 1": []}  # default chat
if "current_chat" not in st.session_state:
    st.session_state["current_chat"] = "Chat 1"

# Sidebar: Manage chats
st.sidebar.header("💬 Chats")
chat_names = list(st.session_state["chats"].keys())
selected_chat = st.sidebar.radio("Select a chat", chat_names, index=chat_names.index(st.session_state["current_chat"]))

# Sync selection with session state
if selected_chat != st.session_state["current_chat"]:
    st.session_state["current_chat"] = selected_chat

# Create new chat
if st.sidebar.button("➕ New Chat"):
    new_chat_name = f"Chat {len(st.session_state['chats']) + 1}"
    st.session_state["chats"][new_chat_name] = []
    st.session_state["current_chat"] = new_chat_name


# Display Current Chat Messages
messages = st.session_state["chats"][st.session_state["current_chat"]]

for msg in messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# Chat Input
if user_input := st.chat_input("Ask me a medical question..."):
    # Add user message
    messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Immediate spinner + response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching medical documents..."):
            response_stream = qa_chain.stream(user_input)  
            bot_reply = st.write_stream(response_stream)
    messages.append({"role": "assistant", "content": bot_reply})


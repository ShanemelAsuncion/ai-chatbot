import streamlit as st
from utils import RAGChatbot
import os

st.title("📚 RAG-Enhanced Chatbot")
st.write("This chatbot uses Retrieval Augmented Generation (RAG) to provide more accurate and contextual responses.")

# Initialize session state for chatbot
if "rag_chatbot" not in st.session_state:
    st.session_state.rag_chatbot = RAGChatbot(
        pinecone_api_key=st.secrets["PINECONE_API_KEY"],
        pinecone_environment=st.secrets["PINECONE_ENVIRONMENT"],
        index_name="ai-chatbot-mlh"
    )

# File uploader for document ingestion
st.sidebar.header("📄 Document Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload documents for the chatbot to learn from",
    accept_multiple_files=True,
    type=["txt", "pdf"]
)

if uploaded_files:
    # Create a temporary directory for uploaded files
    if not os.path.exists("temp_docs"):
        os.makedirs("temp_docs")
    
    # Save uploaded files to temporary directory
    for file in uploaded_files:
        with open(os.path.join("temp_docs", file.name), "wb") as f:
            f.write(file.getbuffer())
    
    if st.sidebar.button("Process Documents"):
        with st.spinner("Processing documents..."):
            st.session_state.rag_chatbot.ingest_documents("temp_docs")
        st.sidebar.success("Documents processed successfully!")

# Chat interface
st.header("💬 Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.rag_chatbot.get_response(prompt)
            st.markdown(response["answer"])
            
            # Display sources if available
            if response["sources"]:
                with st.expander("View Sources"):
                    for i, source in enumerate(response["sources"], 1):
                        st.markdown(f"**Source {i}:**\n{source}\n---")

# Sidebar controls
st.sidebar.header("🔧 Controls")
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.rag_chatbot.clear_history()
    st.sidebar.success("Chat history cleared!")

# Add some helpful information
st.sidebar.header("ℹ️ Information")
st.sidebar.info(
    """
    This chatbot uses RAG (Retrieval Augmented Generation) to provide more accurate responses 
    by referencing your uploaded documents. To use it:
    1. Upload your documents using the file uploader above
    2. Click 'Process Documents' to analyze them
    3. Ask questions about your documents in the chat
    """
)
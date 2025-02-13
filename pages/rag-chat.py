import streamlit as st
from utils import get_document_processor
import tempfile
import os

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize the document processor
processor = get_document_processor()

if not processor:
    st.error("Failed to initialize the chatbot. Please check your API keys and try again.")
    st.stop()

st.title("🤖 RAG-Enhanced Chatbot")
st.markdown("""
This chatbot uses Retrieval Augmented Generation (RAG) to provide accurate answers based on your documents.
Upload PDF documents and ask questions about their content!
""")

# Document upload and processing section
with st.sidebar:
    st.header("📄 Document Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to upload"
    )
    
    if uploaded_files and st.button("Process Documents", key="process_docs"):
        with st.spinner("Processing documents..."):
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files to the temporary directory
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                
                # Process the documents
                success = processor.process_documents(temp_dir)
                if success:
                    st.success("✅ Documents processed successfully!")
                    st.session_state.docs_processed = True
                else:
                    st.error("❌ Failed to process documents. Please try again.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.docs_processed = False

# Chat interface
st.divider()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:**\n{source['text']}\n\n")

# Chat input
if prompt := st.chat_input("Ask a question about your documents...", key="chat_input"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display response
    with st.chat_message("assistant"):
        if not st.session_state.docs_processed:
            st.warning("⚠️ Please upload and process some documents first!")
        else:
            with st.spinner("Thinking..."):
                response = processor.query_documents(prompt)
                if response:
                    st.markdown(response["answer"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"]
                    })
                else:
                    st.error("❌ Sorry, I couldn't find relevant information to answer your question.")

# Sidebar controls
with st.sidebar:
    st.divider()
    st.header("🔧 Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared!")
    
    # Help section
    st.divider()
    st.header("ℹ️ Help")
    st.markdown("""
    **How to use:**
    1. Click 'Browse files' to select PDF documents
    2. Click 'Process Documents' to analyze them
    3. Ask questions in the chat
    4. Click 'View Sources' to see reference text
    """)
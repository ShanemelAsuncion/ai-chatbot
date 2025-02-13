import streamlit as st
from utils import get_document_processor

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
    pdf_dir = st.text_input("PDF Directory Path", value="pdf", help="Enter the path to your PDF documents directory")
    
    if st.button("Process Documents", key="process_docs"):
        with st.spinner("Processing documents..."):
            success = processor.process_documents(pdf_dir)
            if success:
                st.success("✅ Documents processed successfully!")
                st.session_state.docs_processed = True
            else:
                st.error("❌ Failed to process documents. Please check the directory path and try again.")

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
            st.warning("⚠️ Please process some documents first!")
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
    1. Enter the path to your PDF documents
    2. Click 'Process Documents' to analyze them
    3. Ask questions in the chat
    4. Click 'View Sources' to see reference text
    """)
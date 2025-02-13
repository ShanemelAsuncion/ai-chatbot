"""
Core utilities for the RAG (Retrieval Augmented Generation) chatbot system.
This module handles document processing, embedding generation, and vector storage/retrieval.
"""

from typing import List, Dict, Any, Optional
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import hashlib
from functools import lru_cache
import os

class DocumentProcessor:
    """
    Main class for processing documents, generating embeddings, and handling RAG operations.
    Integrates with OpenAI for embeddings/chat and Pinecone for vector storage.
    """
    
    def __init__(self, openai_api_key: str, pinecone_api_key: str, index_name: str):
        """
        Initialize the document processor with necessary API keys and configurations.
        
        Args:
            openai_api_key: API key for OpenAI services
            pinecone_api_key: API key for Pinecone vector database
            index_name: Name of the Pinecone index to use/create
        """

        self.openai_api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Create index if it doesn't exist
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )
        
        self.index = self.pc.Index(index_name)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.index_name = index_name

    @staticmethod
    def chunk_text(text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        Split text into semantically meaningful chunks while preserving paragraph structure.
        This is crucial for RAG as it determines the context windows for retrieval.
        
        Args:
            text: The input text to chunk
            max_chunk_size: Maximum size of each chunk in characters
            
        Returns:
            List of text chunks, each preserving paragraph structure
        """

        if not text.endswith("\n\n"):
            text += "\n\n"
        
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            current_chunk += paragraph.strip() + "\n\n"
            
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    @lru_cache(maxsize=100)
    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for a text chunk with LRU caching for efficiency.
        Uses OpenAI's text embedding model to create vector representations.
        
        Args:
            text: Text to generate embeddings for
            
        Returns:
            List of floating point numbers representing the text embedding
        """

        return self.embeddings.embed_query(text)

    def process_documents(self, directory: str) -> bool:
        """
        Process PDF documents from a directory and store their vector representations in Pinecone.
        
        The process involves:
        1. Loading PDFs from the directory
        2. Chunking the text content
        3. Generating embeddings for each chunk
        4. Storing vectors with metadata in Pinecone
        
        Args:
            directory: Path to directory containing PDF files
            
        Returns:
            bool: True if processing succeeded, False otherwise
        """

        try:
            # Verify directory exists and contains PDF files
            if not os.path.exists(directory):
                st.error(f"Directory {directory} does not exist.")
                return False
            
            pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
            if not pdf_files:
                st.error("No PDF files found in the directory.")
                return False
            
            # Load documents
            loader = PyPDFDirectoryLoader(directory)
            documents = loader.load()
            
            if not documents:
                st.warning("No content found in the PDF files.")
                return False
            
            # Process each document
            all_vectors = []
            total_chunks = 0
            
            for doc in documents:
                # Split into chunks
                chunks = self.chunk_text(doc.page_content)
                total_chunks += len(chunks)
                
                # Generate embeddings and metadata for each chunk
                for chunk in chunks:
                    embedding = self.generate_embeddings(chunk)
                    doc_id = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                    
                    vector_data = {
                        "id": doc_id,
                        "values": embedding,
                        "metadata": {
                            "text": chunk,
                            "source": os.path.basename(doc.metadata.get("source", "unknown"))
                        }
                    }
                    all_vectors.append(vector_data)
            
            # Batch upsert to Pinecone
            if all_vectors:
                batch_size = 100
                for i in range(0, len(all_vectors), batch_size):
                    batch = all_vectors[i:i + batch_size]
                    self.index.upsert(vectors=batch)
                st.info(f"Processed {len(documents)} documents with {total_chunks} chunks.")
                return True
            else:
                st.warning("No content was extracted from the documents.")
                return False
                
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return False

    def query_documents(self, query: str, top_k: int = 3) -> Optional[Dict[str, Any]]:
        """
        Perform RAG query operation to answer questions based on stored documents.
        
        The process:
        1. Generate embedding for the query
        2. Find most similar chunks in Pinecone
        3. Use retrieved context with GPT to generate an answer
        
        Args:
            query: User's question
            top_k: Number of most relevant chunks to retrieve
            
        Returns:
            Dict containing the answer and source information, or None if query fails
        """

        try:
            # Generate query embeddings
            query_embedding = self.generate_embeddings(query)
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            if not results.matches:
                return None
            
            # Combine context from top matches
            context = "\n\n".join([
                f"From {match.metadata['source']}:\n{match.metadata['text']}"
                for match in results.matches
            ])
            
            # Generate response using OpenAI
            completion = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on the given context. Always cite the source document when providing information."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a detailed answer and mention which source document the information comes from."}
                ],
                temperature=0.7
            )
            
            return {
                "answer": completion.choices[0].message.content,
                "sources": [match.metadata for match in results.matches]
            }
            
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return None

@st.cache_resource
def get_document_processor() -> DocumentProcessor:
    """
    Create or retrieve a cached DocumentProcessor instance.
    Uses Streamlit's caching to maintain a single instance across reruns.
    
    Returns:
        DocumentProcessor instance or None if initialization fails
    """

    try:
        return DocumentProcessor(
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            pinecone_api_key=st.secrets["PINECONE_API_KEY"],
            index_name="ai-chatbot-mlh"
        )
    except Exception as e:
        st.error(f"Error initializing document processor: {str(e)}")
        return None
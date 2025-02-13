from typing import List, Dict, Any, Optional
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from pinecone import Pinecone
from openai import OpenAI
import hashlib
from functools import lru_cache

class DocumentProcessor:
    def __init__(self, openai_api_key: str, pinecone_api_key: str, index_name: str):
        """Initialize document processor with necessary API keys and configurations."""
        self.openai_api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.index_name = index_name

    @staticmethod
    def chunk_text(text: str, max_chunk_size: int = 1000) -> List[str]:
        """Split text into chunks of maximum size while preserving paragraph structure."""
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
        """Generate embeddings for a text chunk with caching for efficiency."""
        return self.embeddings.embed_query(text)

    def process_documents(self, directory: str) -> bool:
        """Process documents from a directory and store in Pinecone."""
        try:
            # Load documents
            loader = PyPDFDirectoryLoader(directory)
            documents = loader.load()
            
            # Process each document
            all_vectors = []
            for doc in documents:
                # Split into chunks
                chunks = self.chunk_text(doc.page_content)
                
                # Generate embeddings and metadata for each chunk
                for chunk in chunks:
                    embedding = self.generate_embeddings(chunk)
                    doc_id = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                    
                    vector_data = {
                        "id": doc_id,
                        "values": embedding,
                        "metadata": {
                            "text": chunk,
                            "source": doc.metadata.get("source", "unknown")
                        }
                    }
                    all_vectors.append(vector_data)
            
            # Batch upsert to Pinecone
            if all_vectors:
                batch_size = 100
                for i in range(0, len(all_vectors), batch_size):
                    batch = all_vectors[i:i + batch_size]
                    self.index.upsert(vectors=batch)
                return True
            else:
                st.warning("No documents found in the specified directory.")
                return False
                
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return False

    def query_documents(self, query: str, top_k: int = 3) -> Optional[Dict[str, Any]]:
        """Query documents with error handling and response generation."""
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
            context = "\n\n".join([match.metadata["text"] for match in results.matches])
            
            # Generate response using OpenAI
            completion = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on the given context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
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

# Initialize processor with API keys from Streamlit secrets
@st.cache_resource
def get_document_processor() -> DocumentProcessor:
    """Get or create a DocumentProcessor instance with caching."""
    try:
        return DocumentProcessor(
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            pinecone_api_key=st.secrets["PINECONE_API_KEY"],
            index_name="ai-chatbot-mlh" 
        )
    except Exception as e:
        st.error(f"Error initializing document processor: {str(e)}")
        return None
from typing import List, Dict, Any
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
import pinecone
from dotenv import load_dotenv

"""
Example usage:
    chatbot = RAGChatbot(
    pinecone_api_key="your-api-key",
    pinecone_environment="your-environment",
    index_name="your-index-name"
 )
 chatbot.ingest_documents("path/to/documents")
 response = chatbot.get_response("Your question here")
"""

# Load environment variables
load_dotenv()

class RAGChatbot:
    def __init__(self, pinecone_api_key: str, pinecone_environment: str, index_name: str):
        """Initialize the RAG chatbot with necessary configurations."""
        
        # Initialize Pinecone
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Get or create Pinecone index
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric='cosine'
            )
        
        # Initialize vector store
        self.vectorstore = Pinecone.from_existing_index(
            index_name=index_name,
            embedding=self.embeddings
        )
        
        # Initialize chat model and conversation chain
        self.llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            verbose=True
        )
        
        # Initialize conversation history
        self.chat_history = []

    def ingest_documents(self, directory_path: str) -> None:
        """
        Ingest documents from a directory into the vector store.
        
        Args:
            directory_path: Path to the directory containing documents
        """
        # Load documents
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        # Add documents to vector store
        self.vectorstore.add_documents(texts)

    def get_response(self, query: str) -> Dict[str, Any]:
        """
        Get a response from the chatbot for a given query.
        
        Args:
            query: User's question or prompt
            
        Returns:
            Dictionary containing the response and source documents
        """
        # Get response from conversation chain
        result = self.conversation_chain({
            "question": query,
            "chat_history": self.chat_history
        })
        
        # Update chat history
        self.chat_history.append((query, result["answer"]))
        
        return {
            "answer": result["answer"],
            "sources": [doc.page_content for doc in result["source_documents"]]
        }

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.chat_history = []



# AI Chatbot with RAG

A powerful chatbot application built with Streamlit that combines traditional chat functionality with Retrieval Augmented Generation (RAG) capabilities.

## Features

- **Traditional Chat**:
  - Real-time conversation with GPT models
  - Customizable system prompts
  - Chat history management
  - Modern Streamlit interface

- **RAG-Enhanced Chat**:
  - PDF document upload and processing
  - Context-aware responses based on your documents
  - Source tracking and transparency
  - Real-time document indexing
  - Multiple document support

## Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Pinecone API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-chatbot.git
   cd ai-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API keys:
   Create a `.streamlit/secrets.toml` file with:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key"
   PINECONE_API_KEY = "your-pinecone-api-key"
   ```

4. Run the application:
   ```bash
   streamlit run home.py
   ```

## Usage

### Traditional Chat

1. Navigate to the "Chat" page
2. Type your message in the chat input
3. View the AI's response in the chat history
4. Use the sidebar to:
   - Clear chat history
   - Customize system prompt
   - Adjust model parameters

### RAG Chat

1. Navigate to the "RAG Chat" page
2. Upload PDF documents using the file uploader
3. Click "Process Documents" to analyze them
4. Ask questions about your documents
5. View source references by expanding "View Sources"

## How RAG Works

### Document Processing Pipeline

1. **Document Ingestion**:
   - PDF files are uploaded through the UI
   - Documents are split into manageable chunks
   - Chunk size: 1000 characters (preserves context)

2. **Vector Embedding**:
   - Each chunk is converted to a vector using OpenAI's embedding model
   - Vectors are stored in Pinecone's serverless database
   - Efficient similarity search capabilities

3. **Query Processing**:
   - User questions are converted to vectors
   - Most relevant chunks are retrieved
   - GPT model generates responses using the context

### Technical Architecture

- **Frontend**: Streamlit
- **Vector Database**: Pinecone (Serverless)
- **Embedding Model**: OpenAI Ada
- **Language Model**: GPT-3.5 Turbo
- **Document Processing**: LangChain + PyPDF
- **Vector Search**: Top 3 most relevant chunks

## Security

- API keys are stored securely in `.streamlit/secrets.toml`
- The secrets file is excluded from version control
- Use environment variables in production
- Follow OpenAI and Pinecone security best practices

## Dependencies

- `openai`: OpenAI API client
- `langchain`: LLM framework
- `pinecone-client`: Vector database client
- `streamlit`: Web interface
- `python-dotenv`: Environment management
- `pypdf`: PDF processing
- `tiktoken`: Token counting
- `langchain-community`: Additional LangChain components

## UI

![Home](assets/Screenshot%202025-02-13%20073327.png)

![chat](assets/Screenshot%202025-02-13%20073339.png)

![rag-chat](assets/Screenshot%202025-02-13%20073358.png)


## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Created By

Built with ❤️ by ShanemelAsuncion

## License

This project is licensed under the MIT License - see the LICENSE file for details.
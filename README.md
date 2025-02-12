# AI Chatbot with GPT Integration

A simple and interactive chatbot application built with Streamlit and OpenAI's GPT model. This project demonstrates how to create a user-friendly chat interface that leverages the power of GPT for natural language conversations.

## Features

- Clean and intuitive chat interface
- Real-time streaming responses from GPT
- Session state management for persistent chat history
- Modern UI with Streamlit components
- Support for GPT-3.5-turbo model

## Prerequisites

Before running this application, make sure you have:

- Python 3.x installed
- An OpenAI API key
- Git (optional, for cloning the repository)

## Installation

1. **Clone or Download the Repository**
   ```bash
   git clone https://github.com/ShanemelAsuncion/ai-chatbot.git
   cd ai-chatbot
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv myenv
   myenv\Scripts\activate  # On Windows
   ```

3. **Install Required Packages**
   ```bash
   pip install streamlit openai
   ```

4. **Configure OpenAI API Key**
   - Create a `.streamlit` folder in your project directory
   - Create a `secrets.toml` file inside the `.streamlit` folder
   - Add your OpenAI API key:
     ```toml
     OPENAI_API_KEY = "your-api-key-here"
     ```

## Project Structure

```
ai-chatbot/
├── .streamlit/
│   └── secrets.toml
├── pages/
│   └── chat.py
├── home.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Usage

1. **Start the Application**
   ```bash
   streamlit run home.py
   ```

2. **Access the Chat Interface**
   - Open your web browser
   - Navigate to the local URL provided by Streamlit (typically http://localhost:8501)
   - Use the chat interface to start conversing with the AI

## Security Notes

- Keep your OpenAI API key confidential
- Never commit the `secrets.toml` file to version control
- The `.gitignore` file is configured to exclude sensitive files and directories

## Created By

Built with ❤️ by @ShanemelAsuncion

## License

This project is open source and available under the MIT License.
from openai import OpenAI
import streamlit as st

# Page title
st.title("GPT Wrapper")

# Initialize OpenAI client with API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize session state for model selection
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and response handling
if prompt := st.chat_input("What is up?"):
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        # Create streaming chat completion
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,  # Enable streaming for real-time response
        )
        # Display streaming response and capture final response
        response = st.write_stream(stream)
    # Store assistant response in chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

st.divider()

# Sidebar controls
with st.sidebar:
    st.divider()
    st.header("🔧 Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared!")
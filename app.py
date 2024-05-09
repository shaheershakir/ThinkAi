import streamlit as st

# ... (your existing code for chat functionalities) ...

st.set_page_config(page_title="Langchain Chatbot", page_icon="💬", layout="wide")


def introduction():
    st.title("Welcome to the ThinkAI Chatbot!")
    st.write("Engage in conversations across various media formats!")

    st.header("Key Features:")
    st.markdown(
        """
    *   **Chat with PDFs:** Ask questions and get answers directly from PDF documents. 
    *   **Website Chat:** Interact with website content through conversational queries.
    *   **YouTube Chat:** Discuss and explore YouTube videos through text-based interaction.
    *   **Image Chat:** Ask questions about images and get descriptive responses.
    *   **Audio Chat:** Converse using voice input and receive text-based replies.
    """
    )

    st.header("Technology Behind the Magic:")
    st.markdown(
        """
    This app leverages the power of Natural Language Processing (NLP) and various AI models to understand and respond to your queries in a meaningful way. 
    We utilize libraries like:
    *   **Streamlit:** For building the interactive web application.
    *   **PyPDF2/BeautifulSoup:** For processing PDFs and website content (depending on chosen functionalities).
    *   **NLP Libraries (e.g., transformers, SentenceTransformers):** For understanding and responding to natural language input. 
    *   **(Potentially) Speech Recognition libraries:** For enabling voice-based interaction. 
    """
    )


introduction()

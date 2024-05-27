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
    """
    )

    st.header("Technology Behind the Magic:")
    st.markdown(
        """
1. **Large Language Models (LLMs):**
   - **ChatGPT/Gemini Pro Vision:** These models, developed by OpenAI and Google respectively, form the backbone of these applications. They are capable of understanding and generating human-like text, enabling them to answer questions, summarize information, and even engage in conversations.

2. **Retrieval Augmented Generation (RAG):**
   - **Contextual Retrieval:** Instead of relying solely on the LLM's internal knowledge, RAG enhances the responses by retrieving relevant information from external sources. This ensures more accurate and contextually appropriate answers.

3. **Key Libraries:**
   - **Langchain:** A powerful framework designed to simplify the development of LLM-powered applications. It provides modular components for tasks like data loading, text splitting, embedding generation, vector storage, and chain creation.
   - **Streamlit:** Facilitates the creation of interactive web applications. It allows developers to easily build user interfaces for interacting with the AI models.
   - **Other libraries:** Specialized libraries like youtube-transcript-api for fetching YouTube transcripts and PyPDFLoader for processing PDF files are used for data acquisition and processing.

4. **Embeddings and Vector Stores:**
   - **Embeddings:** Numerical representations of text that capture semantic meaning. Popular embedding models like HuggingFaceEmbeddings and GoogleGenerativeAIEmbeddings are utilized.
   - **Vector Stores:** Databases optimized for storing and searching embeddings. Chroma is a vector store frequently employed in these examples.

5. **Techniques:**
   - **Text Splitting:** Large documents are split into smaller chunks for efficient processing and embedding generation. RecursiveCharacterTextSplitter is commonly used.
   - **RAG-Fusion:** A technique for generating multiple search queries from a user's question and fusing the retrieved results to enhance answer accuracy.

**Functionality Breakdown:**
- **PDF Chat & PDF RAG Function:** These applications allow users to upload PDF documents, ask questions, and receive answers grounded in the content of the uploaded files. RAG-Fusion is employed in one of the examples to improve retrieval quality.
- **Website Chat:** This application allows users to converse with a chatbot that has "read" and understands the information on a given website.
- **YouTube Chat:** Similar to the website chat, this application enables users to ask questions about a specific YouTube video, leveraging its transcript as the knowledge base.
- **Image Chat:** Leverages the capabilities of Gemini Pro Vision to answer questions based on uploaded images, demonstrating the power of multi-modal AI.

**In Essence:**
These code examples highlight the powerful synergy between LLMs, RAG, and various supporting libraries. They demonstrate how AI can be used to access, understand, and interact with information from diverse sources, including documents, websites, videos, and even images.

    """
    )


introduction()

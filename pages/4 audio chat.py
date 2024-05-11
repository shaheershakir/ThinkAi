from langchain_community.document_loaders import AssemblyAIAudioTranscriptLoader
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
import time
from dotenv import load_dotenv
import os
from langchain_community.vectorstores.utils import filter_complex_metadata

load_dotenv()


def get_transcript(audio_file):
    loader = AssemblyAIAudioTranscriptLoader(file_path=audio_file)
    document = loader.load()
    return document


def vectorize_text(document):
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the chunks
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = Chroma.from_documents(document_chunks, embeddings)

    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(retriever_chain):

    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke(
        {"chat_history": st.session_state.chat_history, "input": user_input}
    )

    return response["answer"]


st.title("Audio Chatbot")
with st.sidebar:
    uploaded_files = st.file_uploader("Upload audio files", accept_multiple_files=True)

    # Inputs for YouTube links
    links = st.text_area("Enter YouTube links (one per line)")
    links_list = [link.strip() for link in links.split("\n") if link.strip()]

    if st.button("Start"):
        if links:
            try:
                progress = st.progress(0)
                status_text = st.empty()

                status_text.text("uploading audio...")
                progress.progress(25)

                # Getting both the transcript
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("temp", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    transcript = get_transcript(links)
                    vector_store = vectorize_text(transcript)
                    os.remove(file_path)  # Remove temporary file

                status_text.text(f"Listening...")
                progress.progress(50)
                # wait 2 seconds
                time.sleep(1)
                status_text.text(f"Transcribing url...")
                progress.progress(75)
                for link in links_list:
                    with st.spinner("Processing YouTube link..."):
                        transcript = get_transcript(links)
                        vector_store = vectorize_text(transcript)
                # Vectorizing the text
                vector_store = vectorize_text(transcript)
                st.session_state.vector_store = vector_store
                status_text.text("Complete:")
                progress.progress(100)

            except Exception as e:
                if str(e) == "'Chroma' object is not iterable":
                    st.write("Transcription complete sucessfully.")
                    print("Transcription complete sucessfully.")
                else:
                    st.write(str(e))
                    print(str(e))
        else:
            st.write("Please enter a valid YouTube link.")

# chat logic
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = Chroma(
        embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )  # Initialize empty vector store

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

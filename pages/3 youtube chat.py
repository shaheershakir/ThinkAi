

import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
import datetime
from typing import Literal, Optional, Tuple
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import re
import math
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
)
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import requests
from io import BytesIO

load_dotenv()


def get_transcript(youtube_url):
    try:
        video_id = youtube_url.split("=")[1]
        loader = YoutubeLoader(video_id)
        document = loader.load()
    except NoTranscriptFound:
        raise Exception("No transcript found for this video")

    return document, video_id


def vectorize_text(document):
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the chunks
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
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

st.title("YouTube video summarizer")
with st.sidebar:
    link = st.text_input("Enter the link of the YouTube video you want to summarize:")
    if st.button("Start"):
        if link:
            try:
                progress = st.progress(0)
                status_text = st.empty()

                status_text.text("Getting video...")
                progress.progress(25)

                # Getting both the transcript and video ID
                transcript, video_id = get_transcript(link)

                status_text.text(f"Watching video...")
                progress.progress(50)
                # wait 2 seconds
                time.sleep(2)
                status_text.text(f"Transcriping...")
                progress.progress(75)

                # Vectorizing the text
                vector_store = vectorize_text(transcript)
                st.session_state.vector_store = vector_store
                st.session_state.video_id = video_id  # Store video ID
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
        AIMessage(content="Hello, I am a Think Ai. How can I help you?"),
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = Chroma(
        embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )  # Initialize empty vector store

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    response_text = get_response(user_query) 

    # Display YouTube preview image 
    if "video_id" in st.session_state:
        with st.chat_message("AI"):
            try:
                thumbnail_url = f"https://img.youtube.com/vi/{st.session_state.video_id}/0.jpg"
                print("Thumbnail URL:", thumbnail_url)  # Debugging
                response = requests.get(thumbnail_url)
                image = BytesIO(response.content)
                st.image(image, caption="YouTube Thumbnail", use_column_width=True)
            except Exception as e:
                st.write(f"Error displaying image: {e}")

    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response_text))

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content) 
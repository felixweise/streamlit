import streamlit as st
import requests
import os
# import pinecone
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
#from langchain.vectorstores import Pinecone
from langchain.llms import Replicate
from langchain.prompts import PromptTemplate
from getpass import getpass
# from transformers import pipeline
#from langchain import HuggingFacePipeline


# not used in this case
# Load model directly
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("GermanT5/t5-efficient-gc4-german-base-nl36")
# model = AutoModelForSeq2SeqLM.from_pretrained("GermanT5/t5-efficient-gc4-german-base-nl36")

debug = True
verbose = True

# REPLICATE_API_TOKEN = getpass()

# initialize pinecone
# pinecone.init(api_key="cc43645d-19f7-4efd-a993-66da00a03ecd", environment="gcp-starter")

# index_name = pinecone.Index("instructorxl1")

template = """Sie sind ein hilfsbereiter, respektvoller und ehrlicher Assistent, der Online-Formulare den Benutzern erklärt. Antworten Sie immer so hilfreich wie möglich und gleichzeitig sicher auf Deutsch. Wenn Sie keine Antwort wissen, versuchen Sie nicht eine Antwort zu erfinden.
Befolge die folgenden 2 Schritte:
1. Lies den Kontext unten
2. Beantworte die Frage mit Hilfe der Informationen aus dem Kontext auf Deutsch, benutze kein Englisch.

Kontext : {context}
Nutzerfrage : {question}"""

prompt = PromptTemplate(
    input_variables=["question", "context"], template = template
)

my_prompt= """Sie sind ein hilfsbereiter, respektvoller und ehrlicher Assistent. Antworten Sie immer so hilfreich wie möglich und gleichzeitig sicher. Wenn Sie keine Antwort wissen, versuchen Sie nicht eine Antwort zu erfinden."""

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    #chunks = text_splitter.create_documents(text) - ist für Pinecone notwendig
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    #vectorstore = Pinecone.from_documents(text_chunks, embeddings, index_name = "instructorxl1")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(streaming = True)
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    #llm = HuggingFaceHub(repo_id="GermanT5/t5-efficient-gc4-german-base-nl36", model_kwargs={"temperature":0.75, "max_length":512})
    #llm = HuggingFaceHub(repo_id="jphme/Llama-2-13b-chat-german", model_kwargs={"temperature":0.5, "max_length":512})
    #llm = HuggingFaceHub(repo_id="LeoLM/leo-hessianai-13b-chat", model_kwargs={"temperature":0.75, "max_length":512} )
    #llm = Replicate(
    #verbose=True,
    #model="meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48",
    #model_kwargs={"temperature": 0.75, "max_length": 500, "top_k": 50, "top_p": 1.0},
    #)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True
        #combine_docs_chain_kwargs={"prompt" : prompt}
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat für mehrere PDF-Dokument",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat für mehrere PDF-Dokumente :books:")
    user_question = st.text_input("Stell Deine Fragen zum Inhalt des Dokuments:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Deine Dokumente")
        pdf_docs = st.file_uploader(
            "PDF-Dokumente hochladen und 'Start' klicken", accept_multiple_files=True)
        if st.button("Start"):
            with st.spinner("Analysieren"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
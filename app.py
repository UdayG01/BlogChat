

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3

import streamlit as st

from langchain.document_loaders import UnstructuredURLLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain import PromptTemplate, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import HuggingFaceHub

from htmlTemplates import css, bot_template, user_template

import os
huggingfacehub_api_token = os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


repo_id = "tiiuae/falcon-7b"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, 
                     repo_id=repo_id, 
                     model_kwargs={"max_new_tokens":350})

def get_blog_text(blog_url):
  text = ""
  urls = [blog_url]
  loader = UnstructuredURLLoader(urls=urls)
  data = loader.load()

  for doc in data:
    text += doc.page_content
  return text


def get_text_chunks(text):
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=10)
   docs = text_splitter.split_text(text)
   return docs

def get_vectorstore(docs):
  embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
  db = Chroma.from_texts(docs, embeddings)
  return db

def get_conversation_chain(vectorstore):
  # n_gpu_layers = 40
  # n_batch = 512

  # Loading model,
  # llm = LlamaCpp(
  #     model_path=model_path,
  #     max_tokens=256,
  #     n_gpu_layers=n_gpu_layers,
  #     n_batch=n_batch,
  #     n_ctx=2048,
  #     verbose=False,
  # )

  # repo_id = "google/flan-t5-xxl"
  # llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64})

  memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages=True)
  conversation_chain = ConversationalRetrievalChain.from_llm(
      llm=llm,
      retriever=vectorstore.as_retriever(),
      memory=memory
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
  st.set_page_config(page_title="BlogChat",
                    page_icon=":wind_blowing_face:")
  st.write(css, unsafe_allow_html=True)

  if "conversation" not in st.session_state:
    st.session_state.conversation = None
  if "chat_history" not in st.session_state:
    st.session_state.chat_history = None


  st.header("Chat with Blog :pencil:")
  user_question = st.text_input("Query about your blog ")
  if user_question:
    handle_userinput(user_question)

  with st.sidebar:
    st.subheader("Your URLs")
    with st.form(key='blog-url-form'):
      # Input for the blog URL
      blog_url = st.text_input(label='Enter the blog URL')
      submit_button = st.form_submit_button(label='Submit')

    if submit_button:
      # with st.spinner("Process"):
        # Perform loading on the pdf
        myBar = st.progress(0, text="Processing...")

        text = get_blog_text(blog_url)
        myBar.progress(25, text="Processing...")

        # Perform text-splitting
        docs  = get_text_chunks(text)
        myBar.progress(50, text="Processing...")

        # Performing embeddings/vectorization using HuggingFaceEmbeddings
        # and I'll be storing these embeddings in Chroma vector store
        db = get_vectorstore(docs)
        myBar.progress(75, text="Processing...")

        # Making a conversation chain
        # conversation = get_conversation_chain(db)
        # do the below one in case you want a persistent convo
        st.session_state.conversation = get_conversation_chain(db)
        myBar.progress(100, text="Done")

if __name__ == '__main__':
  main()

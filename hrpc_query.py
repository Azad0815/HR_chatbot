from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import os

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

def build_chat_history(chat_history_list):
    chat_history = []
    for message in chat_history_list:
        chat_history.append(HumanMessage(content=message[0]))
        chat_history.append(AIMessage(content=message[1]))

    return chat_history

def query(question, chat_history):
    chat_history = build_chat_history(chat_history)
    embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
#    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    new_db = FAISS.load_local(r"D:\Learning Materials\CampusX\LangChain\RAG\HR_CHAT_BOT\faiss_index", embedding, allow_dangerous_deserialization=True)
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite",
                                google_api_key=os.getenv("GOOGLE_API_KEY"),
                                temperature=0.5)
    
    condense_question_system_template = (
        "Given a chat history and the latest user question"
        "Which might refrence context in the chat history,"
        "formulate a standalone questions which can be understood"
        "without the chat history. Do NOT answer the question,"
        "just reformulate it if needed and othewise return it as it."
    )

    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")

        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        model, new_db.as_retriever(), condense_question_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks on HR Policy."
        "Use the following pieces of retrieved context to answer"
        "the question. If you don't know the answer, say that you"
        "don't know. "
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(llm=model, prompt=qa_prompt)
    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return convo_qa_chain.invoke(
        {
            "input": question,
            "chat_history": chat_history,
        }
    )


def show_ui():
    st.title("HR Chatbot")
    st.subheader("Please enter your HR Query ")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    # Displaying chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask HR related Query: "):
        # Invoke the function with the Retriver with chat history
        with st.spinner("Working on your query..."):
            response = query(question=prompt, chat_history=st.session_state.chat_history)
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response['answer'])

            # Append user message to chat history
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            st.session_state.messages.append({'role': 'assistant', 'content': response['answer']})
            st.session_state.chat_history.extend([(prompt, response['answer'])])

# Program Entry....
if __name__ == "__main__":
    show_ui()



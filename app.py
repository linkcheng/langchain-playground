from unittest import load_tests
import streamlit as st
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.embeddings import OpenAIEmbeddings
from langchain.experimental import AutoGPT
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.utilities import SerpAPIWrapper
from langchain.vectorstores import Chroma

openai_api_key = st.secrets['openai_api_key']
serpapi_api_key = st.secrets['serpapi_api_key']

def load_vectordb():
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = Chroma(persist_directory='.chroma', embedding_function=embedding)
    return vectordb


def load_tools():
    search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    tools = [
        Tool(
            name = "search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions"
        ),
        WriteFileTool(),
        ReadFileTool(),
    ]
    return tools


def load_llm():
    chat = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name='gpt-3.5-turbo-16k-0613')
    return chat


def load_agent(llm, tools, vectordb):
    agent = AutoGPT.from_llm_and_tools(
        ai_name="Xiaoming",
        ai_role="Assistant",
        tools=tools,
        llm=llm,
        memory=vectordb.as_retriever()
    )
    agent.chain.verbose = True
    return agent


def run(agent, question):
    with get_openai_callback() as cb:
        agent.run(["write a weather report for Paris today"])
        print(cb)

def main():
    agent = load_agent(load_llm(), load_tools(), load_vectordb())
    
    st.title('Auto-GPT')
    with st.container():
        content = st.text_input('How can I help you: ')

    with st.container():
        if content:
            ret = run(agent, content)
            st.text_area('Anwser:', ret)


if __name__ == '__main__':
    main()

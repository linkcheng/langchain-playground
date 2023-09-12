import streamlit as st
from langchain.agents import Tool
from langchain.vectorstores import VectorStore
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.utilities import SerpAPIWrapper


openai_api_key = st.secrets["openai_api_key"]
serpapi_api_key = st.secrets["serpapi_api_key"]
model_name = "gpt-3.5-turbo"


def load_embedding_model():
    return OpenAIEmbeddings(openai_api_key=openai_api_key)


def load_llm():
    chat = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name=model_name)
    return chat


def load_chroma():
    from langchain.vectorstores import Chroma
    
    vectordb = Chroma(persist_directory="./.chroma", embedding_function=load_embedding_model())
    return vectordb


def load_faiss():
    import faiss
    from langchain.vectorstores import FAISS
    from langchain.docstore import InMemoryDocstore
    
    # OpenAI Embedding 向量维数
    embedding_size = 1536
    # 使用 faiss 的 IndexFlatL2 索引
    index = faiss.IndexFlatL2(embedding_size)
    
    embedding = load_embedding_model()
    # 实例化 Faiss 向量数据库
    vectordb = FAISS(embedding.embed_query, index, InMemoryDocstore({}), {})
    return vectordb


def load_tools():
    search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    tools = [
        Tool(
            name="search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        # WriteFileTool(),
        # ReadFileTool(),
    ]
    return tools


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


def search(vectordb: VectorStore, query):
    result = vectordb.similarity_search(query)
    


def main():
    vectordb = load_faiss()
    
    agent = load_agent(load_llm(), load_tools(), vectordb)
    
    st.title("Auto-GPT")

    with st.container():
        content = st.text_input("How can I help you: ")

    with st.container():
        if content:
            placeholder = st.empty()
            placeholder.text("Analyzing ...")
            ret = agent.run([content])
            placeholder.text_area("Anwser: ", ret)


if __name__ == "__main__":
    main()

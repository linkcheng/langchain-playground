import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def translate_prompt():
    sys_template="你是翻译助理，请帮我将以下内容翻译成汉语，谢谢。请只回复翻译文字，不要回复其他内容。"
    sys_message_prompt = SystemMessagePromptTemplate.from_template(sys_template)

    human_template="{content}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    return ChatPromptTemplate.from_messages([sys_message_prompt, human_message_prompt])


def translate(content):
    openai_api_key = st.secrets['openai_api_key']
    chat = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name='gpt-3.5-turbo-16k-0613')
    chain = LLMChain(llm=chat, prompt=translate_prompt(), verbose=True)
    ret = chain.run(content=content)
    return ret


def main():
    st.title('My Translator')

    with st.container():
        content = st.text_area('Text to translate')

    with st.container():
        if st.button('Translate') and content:
            ret = translate(content)
            st.text_area('Anwser:', ret)


if __name__ == '__main__':
    main()

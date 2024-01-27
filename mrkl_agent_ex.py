from contants import COHERE_API_KEY
from langchain_community.chat_models import ChatCohere
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st

prompt=PromptTemplate(
    input_variables=["chat_history","question"],
    template="""You are a kind AI agent, you are currently talking to human answer him/her in a friendly tone 
    
    chat_history:{chat_history}
    human:{question}
    AI: """
)

llm=ChatCohere(api_key=COHERE_API_KEY)
memory=ConversationBufferWindowMemory(memory_key="chat_history",k=5)
llm_chain=LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt
) 
st.set_page_config(
    page_title="Chat with me",
    layout="wide"
)

st.title="Chat with me"

if "messages" not in st.session_state.keys():
    st.session_state.messages=[
        {"role":"assistant","content":"Hello there, ask anything"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_prompt=st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role":"user","content":user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

if st.session_state.messages[-1]["role"]!="assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loding..."):
            ai_response=llm_chain.predict(question=user_prompt)
            st.write(ai_response)
    new_ai_response={"role":"assistant","content":ai_response}
    st.session_state.messages.append(new_ai_response)

with st.sidebar:
    st.header("Chat History")
    with st.expander('conversation'): st.info(memory.buffer)
    

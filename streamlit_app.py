import os
from contants import COHERE_API_KEY
from langchain import PromptTemplate  
from langchain.llms import Cohere
from langchain.chains import LLMChain                                                                                                                                                                                                                                                                                                                                                                                                    

os.environ["COHERE_API_KEY"]=COHERE_API_KEY

import streamlit as st

st.title(':green[Langchain Demo]')
input_text=st.text_input("Search the topic")

template='''In an easy way translate the following sentence'{sentence}' into {target_language}'''
language_prompt= PromptTemplate(
    input_variables=["sentence",'target_language'],
    template=template
)
language_prompt.format(sentence=input_text,target_language='urdu')
llm=Cohere(model="command", temperature=0.75)
chain1=LLMChain(llm=llm,prompt=language_prompt)


if input_text:
    st.write(chain1({'sentence':input_text, 'target_language':'urdu'}))
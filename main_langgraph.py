from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Literal, TypedDict
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


modelo = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=api_key
)

prompt_consultor_de_praia = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sra. Praia. Você é uma especialista em viagens com destinos para praia"),
        ("human", "{query}")
    ]
)

prompt_consultor_de_montanha = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sra. Montanha. Você é uma especialista em viagens com destinos para montanhas e atividades radicais"),
        ("human", "{query}")
    ]
)

cadeia_praia = prompt_consultor_de_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_de_montanha | modelo | StrOutputParser()

class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "responda apenas com 'praia' ou 'montanha'"),
        ("human", "{query}")
    ]
)

roteador = prompt_roteador | modelo.with_structured_output(Rota)

def responde(pergunta : str):
    rota = roteador.invoke({
        "query": pergunta
    })["destino"]
    print(rota)
    
    if rota == "praia":
        return cadeia_praia.invoke({"query": pergunta})
    return cadeia_montanha.invoke({"query": pergunta})

print(responde("Quero passear por praias belas no Brasil"))
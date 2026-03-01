from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
numero_dias = 7
numero_criancas = 2
atividade = "praia"

mensagens = [
    SystemMessage(content="Você é um estagiario feliz e esta com muita vontade de responder as pessoas. voce trabalha em uma agencia de viagens. responda sempre da melhor maneira possivel e com muitos emojis"),
    HumanMessage(content=f"Crie um roteiro de viagens para {numero_dias} dias para uma família com {numero_criancas} crianças que busca atividades relacionadas a {atividade}.")
]
modelo = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key
)

resposta = modelo.invoke(mensagens)
print(resposta.content)
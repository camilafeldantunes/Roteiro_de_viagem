import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


modelo = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key
)

prompt_sugestao= ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um guia de viagens especializado em destinos brasileiros. Apresente-se como Srta. Alegria"),
        ("placeholder", "{historico}"),
        ("human", "{query}")
    ]
)

cadeia = prompt_sugestao | modelo | StrOutputParser()

memoria = {}
sessao = "langchain_python"

def historico_por_sessao(sessao : str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

perguntas = [
    "Quero visitar um lugar no Brasil famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?"
]

#atualiza a memória, responde a pergunta com base no histórico e segue o que a cadeia passou
cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia, #define a cadeia que será executada, especificando o fluxo de processamento
    get_session_history=historico_por_sessao, #recupera o histórico da conversa atual para manter o contexto
    input_messages_key="query", #define a chave que identifica a entrada do usuário na mensagem
    history_messages_key="historico" #define a chave onde o histórico da conversa será armazenado e atualizado
)

for uma_pergunta in perguntas:
    resposta = cadeia_com_memoria.invoke(
        {
            "query": uma_pergunta
        },
        config={"session_id": sessao}
    )
    print("Usuário: ", uma_pergunta)
    print("Chat: ", resposta, "\n")
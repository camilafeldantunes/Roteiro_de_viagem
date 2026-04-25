from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

modelo = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=api_key
)


embeddings = OpenAIEmbeddings()


url = "https://www.todamateria.com.br/voleibol/"
not_volei = WebBaseLoader(web_path=url).load() ## url carregando


pedacos = RecursiveCharacterTextSplitter(
    chunk_size = 1000, chunk_overlap=100
).split_documents(not_volei) ##separa a pagina em pedacos e 1000 caracteres e com uma redundancia de 100

dados_recuperados = FAISS.from_documents(
    pedacos, embeddings
).as_retriever(search_kwargs={"k":2}) 


prompt_cartao_consulta_seguro = ChatPromptTemplate.from_messages(
    [
        ("system", "responda usando exclusivamente o conteúdo fornecido"),
        ("human", "{query}. \n\n Contexto: {contexto}.\n\n Resposta: ")
    ]
)

cadeia = prompt_cartao_consulta_seguro | modelo | StrOutputParser()

def responder(pergunta:str):
    trechos = dados_recuperados.invoke(pergunta) ##chama a cadeia para separar os trechos mais parecidos com a pergunta
    contexto = "\n\n".join(um_trecho.page_content for um_trecho in trechos) ## o contexto é os trechos que foram filtrados
    return cadeia.invoke(
        {
            "query": pergunta, "contexto":contexto
        }
    )

print(responder("O que é vôlei? e porque as pessoas ficam na quadra?"))

print("="*50)

print(not_volei)
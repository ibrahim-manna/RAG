import os
from operator import itemgetter
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

load_dotenv()

TEMPLATE = """
Answer the question based on the page content in the context below. Make sure to check the source of the information in the metadata to ensure you pick the correct page_content. If you can't answer the question, reply with only "Oof that's a tough one, i don't really know this"

Context : {context}

Question : {question}

"""


class RAGModel:
    def __init__(self, db: FAISS):
        self.db = db
        self.llm = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        self.prompt = PromptTemplate.from_template(TEMPLATE)
        self.retriever = db.as_retriever()
        self.chain = (
            {
                "context": itemgetter("question") | self.retriever,
                "question": itemgetter("question"),
            }
            | self.prompt
            | self.llm
        )

    def ask(self, question: str) -> str:
        """Run a question through the RAG chain and return the answer."""
        return self.chain.invoke({"question": question})
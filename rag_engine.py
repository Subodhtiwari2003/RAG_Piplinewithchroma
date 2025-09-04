from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chromadb", embedding_function=embeddings)

    llm = HuggingFaceHub(
        repo_id="tiiuae/falcon-7b-instruct",
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        model_kwargs={
            "temperature": 0.5,
            "max_new_tokens": 512,
            "max_length": 64
        }
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
    return qa_chain

def get_answer(query: str) -> str:
    qa_chain = load_rag_pipeline()
    return qa_chain.run(query)
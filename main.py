from fastapi import FastAPI, Request
from pydantic import BaseModel
from rag_engine import get_answer

app = FastAPI(title="RAG Pipeline with ChromaDB")

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_rag(request: QueryRequest):
    answer = get_answer(request.question)
    return {"question": request.question, "answer": answer}
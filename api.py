from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_helper import get_qa_chain, create_vector_db

app = FastAPI()

# Add CORS so your Node server can talk to it easily
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(request: ChatRequest):
    print(f"Received question: {request.question}") # Debugging line
    chain = get_qa_chain()
    
    if chain is None:
        return {"answer": "Knowledgebase not found. Please click 'Update Knowledgebase' first."}
    
    try:
        response = chain.invoke({"query": request.question})
        return {"answer": response["result"]}
    except Exception as e:
        print(f"Error: {e}")
        return {"answer": f"AI Error: {str(e)}"}

@app.post("/update-db")
def update_db():
    create_vector_db()
    return {"status": "success"}
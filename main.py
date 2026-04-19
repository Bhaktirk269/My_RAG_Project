import os
from fastapi import FastAPI, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
# Use langchain_classic for RetrievalQA in this installed version of LangChain
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
app = FastAPI()

# Add CORS middleware to allow frontend requests
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Use the free local models you downloaded
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Save PDF locally
    file_path = f"./{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Load and split
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)
    
    # Save to local database
    Chroma.from_documents(chunks, embeddings, persist_directory="./free_db")
    return {"status": "Successfully learned the document for free!"}

@app.post("/ask")
async def ask(question: str):
    db = Chroma(persist_directory="./free_db", embedding_function=embeddings)
    
    # This connects the local Llama3 to your PDF data
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )
    
    response = qa_chain.invoke(question)
    return {"answer": response["result"]}
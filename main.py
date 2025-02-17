from langchain_ollama import OllamaLLM,OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from prompt_template import *
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os,uuid

pdf_path="iphone_manual.pdf"
llm=OllamaLLM(model="llama3")

def load_documents(pdf_path):
    loader=PyPDFLoader(pdf_path)
    return loader.load()

#chunking
def split(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100,length_function=len)
    
    chunks=text_splitter.split_documents(document)
    
    return chunks

#initialize a vector DB; returns an instance of chromadb
def init_db():
    embeddings=OllamaEmbeddings(model="nomic-embed-text")

    collection_name=str(uuid.uuid4()) 

    return Chroma(
        persist_directory="./chroma_langchain_db",
        embedding_function=embeddings,
       collection_name=collection_name 
    ) 
    
def doc2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#Generate the embeddings of the chunks
def create_embeddings(chunks,db):
    for chunk in chunks:
        document = Document(page_content=chunk.page_content,  )
        db.add_documents([document])


def RAG(instruction):
    db=init_db()
    document=load_documents(pdf_path)   
    chunks=split(document)
    create_embeddings(chunks,db)

    retriever=db.as_retriever(search_kwargs={"k":2})
    context_text=doc2str(retriever.invoke(instruction))


    prompt=PromptTemplate.from_template(template())
    chain=prompt | llm 

    response=chain.invoke({"instruction":instruction,"context": context_text})
    print(response)


instruction="what is the document about?"
RAG(instruction)
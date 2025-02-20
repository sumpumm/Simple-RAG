from ragatouille import RAGPretrainedModel
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
from prompt_template import *
import uuid

pdf_path = "manual.pdf"

llm = OllamaLLM(model="llama3", temperature=1)

def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

# Chunking
def split(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_documents(document)
    return chunks

# Initialize a vector DB; returns an instance of chromadb
def init_db():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    collection_name = str(uuid.uuid4())
    return Chroma(
        persist_directory="./chroma_langchain_db",
        embedding_function=embeddings,
        collection_name=collection_name
    )

# Generate the embeddings of the chunks
def create_embeddings(chunks, db):
    for chunk in chunks:
        document = Document(page_content=chunk.page_content, metadata=chunk.metadata)
        db.add_documents([document])

def doc2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rerank_with_colbert(query, documents):
    # Initializepre-trained ColBERT model
    colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    
    document_texts = [doc.page_content for doc in documents]
    
    # Rerank the documents
    reranked_results = colbert.rerank(query, document_texts, k=len(documents))
    # return reranked_results
    # Extract the indices from the reranked results
    reranked_indices = [result["result_index"] for result in reranked_results]
    return [documents[i] for i in reranked_indices]

def RAG(question):
    loaded_document = load_documents(pdf_path)
    chunks = split(loaded_document)
    db = init_db()
    # BM25 retrieval
    BM_retriever = BM25Retriever.from_documents(chunks)
    BM_retriever.k = 5
    BM_results = BM_retriever.invoke(question)

    # Vector search
    create_embeddings(chunks, db)
    vector_retriever = db.as_retriever(search_kwargs={"k": 5})
    vector_results = vector_retriever.invoke(question)

    results = BM_results + vector_results

    reranked_results = rerank_with_colbert(question, results)
    
    page_numbers = [reranked_result.metadata.get("page") for reranked_result in reranked_results[:5]]
   
    context_text=doc2str(reranked_results[:5])
    prompt=PromptTemplate.from_template(template())
    chain=prompt | llm 

    response=chain.invoke({"instruction":question,"context": context_text})
    print(response,"\n sources: ",page_numbers)


while True:
    question=input("Ask away: ")
    print(RAG(question))
    
    choice=input("\nEnter 1 to continue and 0 to exit: ")
    if choice == "0":
        break
        
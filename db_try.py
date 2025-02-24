import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

pdf_path = "iphone_manual.pdf"
client = chromadb.PersistentClient(path="./chroma_db")  
collection = client.get_or_create_collection(name="collection_name")

def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

# Chunking
def split(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len)
    return text_splitter.split_documents(document)

def create_embeddings(chunks):
    ids = []
    documents = []  # Store raw text
    metadatas = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_text = chunk.page_content  
        chunk_metadata=chunk.metadata
        ids.append(str(i))  
        documents.append(chunk_text)  
        metadatas.append(chunk_metadata)  
    # Add to ChromaDB
    collection.add(
        ids=ids,  
        documents=documents,
        metadatas=metadatas
    )

def vector_retriever(question):
    results = collection.query(query_texts=[question], n_results=5)
    page_content=[doc for sublist in results['documents'] for doc in sublist]
    page_metadata=[meta for sublist in results['metadatas'] for meta in sublist]

    document=[]
    for x,y in zip(page_content,page_metadata):
        document.append(Document(page_content=x, metadata=y))
        
    return document
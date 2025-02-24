from ragatouille import RAGPretrainedModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
from prompt_template import *
from hallucination_checker import evaluator
import uuid,chromadb

pdf_path = "manual.pdf"
client = chromadb.PersistentClient(path="./chroma_db")  
collection_name=str(uuid.uuid4())
collection = client.get_or_create_collection(name=collection_name)
llm = OllamaLLM(model="llama3", temperature=1)
SIMILARITY_THRESHOLD=0.7

def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

# Chunking
def split(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_documents(document)
    return chunks

# Generate the embeddings of the chunks
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
    
def retrieve_from_all_collections(question):
    collections = client.list_collections()  # List all collections
    for collection in collections:
        # Get the collection object by name
        col = client.get_collection(collection)
        results = col.query(query_texts=[question], n_results=5)
        page_content=[doc for sublist in results['documents'] for doc in sublist]
        page_metadata=[meta for sublist in results['metadatas'] for meta in sublist]
        similarity_score=[score for sublist in results['distances'] for score in sublist]

        document=[]
        for x,y,z in zip(page_content,page_metadata,similarity_score):
            if z>=SIMILARITY_THRESHOLD:
                y["similarity_score"] =z
                document.append(Document(page_content=x, metadata=y))
    return document

def doc2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rerank_with_colbert(query, documents):
    # Initializepre-trained ColBERT model
    colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    
    document_texts = [doc.page_content for doc in documents]
    
    # Rerank the documents
    reranked_results = colbert.rerank(query, document_texts, k=5)

    # Extract the indices from the reranked results
    reranked_indices = [result["result_index"] for result in reranked_results]
    return [documents[i] for i in reranked_indices]

def main(question,chunks):
    # BM25 retrieval
    BM_retriever = BM25Retriever.from_documents(chunks)
    BM_retriever.k = 5
    BM_results = BM_retriever.invoke(question)

    # Vector search
    vector_results = retrieve_from_all_collections(question)
    
    results = BM_results + vector_results

    reranked_results = rerank_with_colbert(question, results)
    print(reranked_results)
    page_numbers = [reranked_result.metadata.get("page") for reranked_result in reranked_results]
   
    context_text=doc2str(reranked_results)
    prompt=PromptTemplate.from_template(template())
    chain=prompt | llm 

    response=chain.invoke({"instruction":question,"context": context_text})
    print(response,"\n sources: ",page_numbers)
    
    context_list=[context_text]
    evaluator(question,context_list,response)



loaded_document = load_documents(pdf_path)
chunks = split(loaded_document)
create_embeddings(chunks)

while True:
    question=input("Ask away: ")
    print(main(question,chunks))
    
    choice=input("\nEnter 1 to continue and 0 to exit: ")
    if choice == "0":
        break



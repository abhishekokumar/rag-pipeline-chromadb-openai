import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def main():
    print("Starting ingestion pipeline...")
    
    # Load documents
    documents = load_doc(docs_path="docs")
    
    # Split documents into chunks
    chunks = split_doc(documents, chunk_size=800)
    
    # store chunks into vector db
    vector_store_embeddings = save_vector_embedding(chunks)
    
    
    
    
def load_doc(docs_path):
    # check if doc directory exists:
    print(docs_path)
    print(os.path.exists(docs_path))
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")
    
    loader = DirectoryLoader(
        path = docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
        )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No documents found in the directory {docs_path}.")
    
    for i, doc in enumerate(documents[:2]):
        print(f'\n Document_ {i+1} loaded with length')
        print(f'Sources_ {i}: {doc.metadata["source"]}')
        print(f'Content first 100 chars {i}: {doc.page_content[:400]}...')  # Print first 100 characters
        #print(f' metadata_ {doc}')
        print("------------------------------------------------------------------")
    
    return documents

def split_doc(documents, chunk_size= 1000, chunk_overlap = 0):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f'\n Document chunk {i+1} with length {len(chunk.page_content)}')
            print(f'Sources_ {i}: {chunk.metadata["source"]}')
            print(f' Length: {len(chunk.page_content)} characters')
            print(chunk.page_content)
            print("------------------------------------------------------------------")

        if len(chunks) > 5:
            print(f" \n ... and {len(chunks) - 5} more chunks.")
    return chunks

def save_vector_embedding(chunks, persist_directory = 'db/chroma_db'):
    embedding_model = OpenAIEmbeddings(model = 'text-embedding-3-small')
    
    #Create chroma DB Vector Store
    print('-- Creating vector store')
    
    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = persist_directory,
        collection_metadata = {'hnsw:space': 'cosine'}
    )
    
    print(' --- Vector Store creation Done ---')
    print(f' Vector Store created and saved to {persist_directory}')
    return vector_store


if __name__ == "__main__":
    main()
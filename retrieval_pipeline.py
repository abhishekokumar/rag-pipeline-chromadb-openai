from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

#Loading embeddings from the persisted directory

embeddings_model = OpenAIEmbeddings(model= "text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings_model,
    collection_metadata={'hnsw:space':'cosine'}
)

#searh for relevant documents
# query = 'which island does SpaceX lease for its launch in the pacific?'
#query = 'In What year did tesla begin Production of the Roadster?'
#query = 'What was NVIDIA\'s first graphics accelerator called?'
#query = ' How much microsoft paid to acquire github?'
query = 'What other companies come under Alphabet Inc besides Google?'

retriever = db.as_retriever(search_kwargs={"k":3})

relevant_docs  = retriever.invoke(query)

print(f'User Query: {query}')

#Display results
print(f'-- Context ---')

for i, doc in enumerate(relevant_docs, 1):
    print(f'Documents {i}:\n {doc.page_content} \n')
    
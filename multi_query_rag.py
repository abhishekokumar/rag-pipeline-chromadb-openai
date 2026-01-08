from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from collections import defaultdict

load_dotenv()

persistent_dir = 'db/chroma_db'
embedding_model = OpenAIEmbeddings(model = 'text-embedding-3-small')
llm = ChatOpenAI(model = 'gpt-4o', temperature = 0)

db = Chroma(
    persist_directory = persistent_dir,
    embedding_function = embedding_model,
    collection_metadata = {'hnsw:space':'cosine'}
)

# pydantic model for structure output
class QueryVariations(BaseModel):
    queries: List[str]
    
# Original query
original_query = 'How does Tesla make money?'
print(f'Original Query: {original_query} \n')

# Generate multiple query variations
llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f""" Generate 3 different variations of this query that would hep retrieve relevant documents:
Original query: {original_query}
Return 3 alternative queries that rephrase or approach the smae question from different angles"""

response = llm_with_tools.invoke(prompt)
query_variations = response.queries

print("Generate Query Variations: ")
for idx, variation in enumerate(query_variations, 1):
    print(f'Variation {idx}: {variation}')
    
print('\n' + '='*60)

# retrieve document with query variation
retriever = db.as_retriever(search_kwargs = {'k':5})
all_retrieval_results = []

for i, query in enumerate(query_variations, 1):
    print(f'\n === RESULTS FOR QUERY {i}: {query} ===')
    docs = retriever.invoke(query)
    all_retrieval_results.append(docs) # store for RRF Calculation
    
    print(f'Retrieved {len(docs)} documents:\n')
    
    for j, doc in enumerate(docs, 1):
        print(f'Documents {j}:')
        print(f'Document {j}: {doc.page_content[:150]} \n')
    
    print('-'*60)
    
print('\n'+'='*60)
print('Multi-Query Retrieval Results completed.')


# ------------------------- Reciprocal Rank Fusion (RRF) -------------------------
def reciprocal_rank_fusion(chunk_lists, k=60, verbose = True):
    if verbose:
        print('\n' + '='*60)
        print('Applying Reciprocal Rank Fusion')
        print('='*60)
        print(f'\n using k = {k}')
        print('calculating RRF Score\n')
        
    # Data Structure for RRF calculation
    rrf_scores = defaultdict(float) # will store : chunk contenct -> rrf score
    all_unique_chunks = {} # will store : chunk content -> actual chunk object
    
    # for verbose output - track chunk id
    chunk_id_map = {}
    chunk_counter = 1
    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"Processing Query {query_idx} results:")
        
        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier
            chunk_content = chunk.page_content
            
            # Assign a simple ID if we haven't seen this chunk before
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            
            chunk_id = chunk_id_map[chunk_content]
            
            # Store the chunk object (in case we haven't seen it before)
            all_unique_chunks[chunk_content] = chunk
            
            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)
            
            # Add to RRF score
            rrf_scores[chunk_content] += position_score
            
            if verbose:
                print(f"  Position {position}: {chunk_id} +{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})")
                print(f"    Preview: {chunk_content[:80]}...")
        
        if verbose:
            print()
    
    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )
    
    if verbose:
        print(f"✅ RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries.")
    
    return sorted_chunks
    
rrf_result = reciprocal_rank_fusion(all_retrieval_results, k = 60, verbose = True)

print('\n' + '='*60)
print(f"\nTop {min(10, len(rrf_result))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(rrf_result[:10], 1):
    print(f"🏆 RANK {rank} (RRF Score: {rrf_score:.4f})")
    print(f"{doc.page_content[:200]}...")
    print("-" * 50)

print(f"\n✅ RRF Complete! Fused {len(rrf_result)} unique documents from {len(query_variations)} query variations.")
    
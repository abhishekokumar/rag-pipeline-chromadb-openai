from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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

# Create chat model
model = ChatOpenAI(model="gpt-4o")

# storing history previous questions and answers for history aware retrieval
chat_history = []

def ask_question(user_question: str):
    '''Function to ask question and get answer from the vector store with context retrieval'''
    print(f'-- User Question: {user_question}')
    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, Rewrite the new question to be standalone and searchable. just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f" New question: {user_question}")
        ]
        
        result = model.invoke(messages)
        searchable_question = result.content.strip()
        print(f'-- Rewritten Searchable Question for retrieval: {searchable_question}')
    else:
        searchable_question = user_question
    #----------------------------------------------
    # simple retrieval method without any threshold
    #----------------------------------------------
    #retriever = db.as_retriever(search_kwargs={"k": 3}) 
    
    #--------------------------------------
    # similarity threshold retrieval method
    #--------------------------------------
    # retriever = db.as_retriever(
    #     search_type = 'similarity_score_threshold',
    #     search_kwargs={
    #         "k": 3,
    #         "score_threshold": 0.4})
    #----------------------------------------
    # max marginal relevance retrieval method
    #----------------------------------------
    retriever = db.as_retriever(
        search_type = 'mmr',
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.5})
    relevant_docs = retriever.invoke(searchable_question)
    
    #Display results
    print(f'-- Context ---')
    for i, doc in enumerate(relevant_docs, 1):
        print(f'Documents {i}:\n {doc.page_content} \n') 
    
    # Combine query and context to generate answer
    combine_input = f"""Based on the following documents, please answer this question: {user_question}
    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
    Please provide a clear , helful answer using only the information from these documents. If you can't find the answer in the documents, please say "We unable to answer this question.".
    """
    
    # Define the messages for the model
    messages = [
        SystemMessage(content="You are a helpful assistant that provides answers based on the provided documents and conversation."),
        HumanMessage(content=combine_input)
    ]
    
    # Get the model's response
    result = model.invoke(messages)

    # Display the answer
    print(f'-- Generated Answer: \n{result.content}')
    response_answer = result.content
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=response_answer))
    print(f" Answer: {response_answer}")
    return response_answer

def start_chat():
    print('Ask me question! Type "quit" to exit.')

    while True:
        question = input("\n your question: ")
        if question.lower() == "quit":
            print("Exiting the chat. Goodbye!")
            break
        ask_question(question)
        
if __name__ == "__main__":
    start_chat()
        
#searh for relevant documents
# query = 'which island does SpaceX lease for its launch in the pacific?'
#query = 'In What year did tesla begin Production of the Roadster?'
#query = 'What was NVIDIA\'s first graphics accelerator called?'
#query = ' How much microsoft paid to acquire github?'
#query = 'What other companies come under Alphabet Inc besides Google?'
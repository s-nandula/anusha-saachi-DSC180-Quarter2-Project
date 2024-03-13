from utils import initialize_embeddings, initialize_vectordb, load_llm_model, initialize_pipeline, initialize_qa
from constants import *
from langchain.llms import HuggingFacePipeline

def main():
    embeddings = initialize_embeddings()
    vectordb = initialize_vectordb(embeddings)
    
    model, tokenizer = load_llm_model()
    pipeline = initialize_pipeline(model, tokenizer)
    
    llm = HuggingFacePipeline(pipeline=pipeline)
    retriever = vectordb.as_retriever()
    
    qa = initialize_qa(llm, retriever)
    
    # Ask the user to input their query
    print("Please enter your query:")
    user_query = input()
    instruction = " Please provide a detailed answer that is based solely on the context provided."
    
    # Combine user input with the instruction
    query = user_query + instruction
    
    print(f"\nPlease be patient. This may take a few minutes!\n")
    result = qa.run(query)
    print("\nResult: ", result)

if __name__ == "__main__":
    main()
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
import torch

from constants import MODEL_PATH, MODEL_DIR, PERSIST_DIRECTORY, MODEL_KWARGS

def initialize_embeddings():
    return SentenceTransformerEmbeddings(model_name=MODEL_PATH, model_kwargs=MODEL_KWARGS)

def initialize_vectordb(embeddings):
    return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

def load_llm_model():
    model = LlamaForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True)
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer

def initialize_pipeline(model, tokenizer):
    """
    Initialize the pipeline for text generation with specific configurations.

    Args:
        model: The LLM model.
        tokenizer: Tokenizer for the model.

    Returns:
        A text generation pipeline.
    """
    pipeline_kwargs = {
         "torch_dtype": torch.float16,
        "device_map": "auto",
        "temperature": 0.8,
    }
    
    return transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
         **pipeline_kwargs
    )

def initialize_qa(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        verbose=True
    )
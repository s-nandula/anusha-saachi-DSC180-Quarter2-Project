# constants.py

# Model and tokenizer directories
MODEL_PATH = "./multi-qa-MiniLM-L6-cos-v1/"
MODEL_DIR = 'Llama-2-7b-chat-hf/'

# Persistence directory for the vector database
PERSIST_DIRECTORY = "./chroma_db"

# Model kwargs for SentenceTransformerEmbeddings and pipeline settings
MODEL_KWARGS = {"device": "cuda"}


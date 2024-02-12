#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3
import os


documents = read_documents(directory)
create_database(db_name)


# In[2]:


import sqlite3

# Replace 'your_database.db' with the path to your SQLite database file, in our case our data is in a protected envionrment so we can not upload it to this github.
db_path = 'documents.db'
# Replace 'table_name' with the name of your table, ours is documents
table_name = 'documents'

# Connect to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Execute the query to count rows in the table
query = f"SELECT COUNT(*) FROM {table_name}"
cursor.execute(query)

# Fetch the result
rows_count = cursor.fetchone()[0]

print(f"Number of rows in {table_name}: {rows_count}")

# close the connection
conn.close()


# In[3]:


import sqlite3
import numpy as np
import faiss
import torch
import openai
from transformers import BertTokenizer, BertModel
from concurrent.futures import ThreadPoolExecutor


# In[4]:

#function to fetch document ids
def fetch_document_ids(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM documents")  # Adjust based on your table schema
    ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return ids


# In[5]:

#function to fetch document text by id
def fetch_document_by_id(db_path, doc_id):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM documents WHERE id = ?", (doc_id,))
        result = cursor.fetchone()
    return result[0] if result else None


# In[6]:

#paralleizing document retrieval
def fetch_documents_parallel(db_path, document_ids):
    with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers as needed
        documents = list(executor.map(lambda doc_id: fetch_document_by_id(db_path, doc_id), document_ids))
    conn.close()
    return documents


# In[7]:


# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


# In[8]:


# Batch encoding function
def encode(texts, batch_size=32):
    model.eval()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings.cpu())
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings.numpy()


# In[9]:


# quantized FAISS index
def create_quantized_faiss_index(encoded_docs):
    dimension = encoded_docs.shape[1]
    nlist = min(len(encoded_docs), 2048)
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 256, 8) 
    assert not index.is_trained
    index.train(encoded_docs)
    index.add(encoded_docs)
    return index


# In[10]:


def create_quantized_faiss_index_chunks(encoded_docs):
    dimension = encoded_docs.shape[1] 
    index = faiss.IndexFlatL2(dimension)  
    
    if encoded_docs.dtype != np.float32:
        encoded_docs = encoded_docs.astype(np.float32)
    
    index.add(encoded_docs) 
    return index


# In[90]:

#search function for the query
def search_faiss(query, faiss_index, k=5):
    faiss_index.nprobe = 10
    encoded_query = encode(query)
    distances, indices = faiss_index.search(encoded_query, k)
    return [i for i in indices[0] if i < len(documents)]


# In[12]:

#chunking function for the nearest neighbors


def chunk_document(document, char_limit=500):
    # Tokenize the document
    tokens = tokenizer.tokenize(document)
    # Convert tokens back to strings to better estimate actual character lengths including spaces
    token_strs = tokenizer.convert_tokens_to_string(tokens).split(' ')
    
    chunked_texts = []
    current_chunk = ""
    
    for token_str in token_strs:
        new_chunk_length = len(current_chunk) + len(token_str) + 1  
        
        if new_chunk_length <= char_limit:
            current_chunk = f"{current_chunk} {token_str}".strip()
        else:
            chunked_texts.append(current_chunk)
            current_chunk = token_str 
            
    if current_chunk:
        chunked_texts.append(current_chunk)
    
    return chunked_texts

# Function to get top N documents, chunk them, and prepare for LLM
def get_top_documents_and_chunk(top_chunk_ids):
    top_documents = [documents[doc_id] for doc_id in top_chunk_ids]
    chunked_documents = []
    for doc in top_documents:
        chunked_documents.extend(chunk_document(doc))
    return chunked_documents


# In[13]:

#From here down is a usecase
db_path = 'documents.db'
document_ids = fetch_document_ids(db_path)


# In[14]:


documents = fetch_documents_parallel(db_path, document_ids)


# In[15]:


encoded_docs = encode(documents)


# In[16]:


faiss_index = create_quantized_faiss_index(encoded_docs)


# In[81]:


#query = "What is the positive impact of generative ai"
query = "Rivian"


# In[87]:


# In[91]:


top_documents_ids = search_faiss(query, faiss_index, k=5)


# In[92]:


top_documents_ids


# In[89]:


documents[331]


# In[72]:


openai.api_key = 'YOUR API KEY (ours is not allowed to be posted)'


# In[73]:


chunks = get_top_documents_and_chunk(top_documents_ids)


# In[74]:


encoded_docs_chunks = encode(chunks)


# In[75]:


faiss_index_chunks = create_quantized_faiss_index_chunks(encoded_docs_chunks)


# In[76]:


best_chunks = [chunks[i] for i in search_faiss(query, faiss_index_chunks, 8)]


# In[77]:


chunks


# In[78]:


def generate_answer(top_chunks, query):
    prompt = f"Question: {query}\n\nContext:\n"
    for chunk in top_chunks:  # Adjust as needed
        prompt += chunk + "\n\n"
    response = openai.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": prompt}
  ]
)
    return response.choices[0].message.content


# In[79]:


answer = generate_answer(best_chunks, query)


# In[80]:


answer


# In[ ]:





# In[ ]:





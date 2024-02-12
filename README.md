# Overview 

This Quarter 2 Project attempts to address historical information retrieval by integrating Retrieval Augmented Generation (RAG) and Large Language Models (LLMs) with Semanitc Search. It aims to overcome the limitations of conventional search methods by searching from a range of historical news sources to ensure the credibility and relevance of the information provided.






# Project Structure
The repository is organized as follows:
```
anusha-saachi-DSC180-Quarter2-Project/
├─ search.py
├─ End-to-End--Database-Creation.py
├─ README.md

```

# Usage

## Run Locally: 
Please Note: If you wish to run this locally, it will not run unless you have your own SQLite database/ documents. This code is specifically designed to work on the non-exportable data in TDM studios. Also the dependencies and structure of the code are hevaily influenced by the limitations and accessible tools of the TDM studios virtual envionrment.

1. Clone this repository on your local machine
2. Open your terminal
3. Change (cd) into the directory to the cloned repository
4. Make sure all the necessary packages for running the project all installed
5. Use End-to-End--Database-Creation.py to create your sqlite table
6. Use search.py to execute the search algorithm, replace "query" with the query you would like to run the search algorithm on.


## Requirements
1) Python 3
2) Libraries listed in search.py and End-to-End--Database-Creation.py

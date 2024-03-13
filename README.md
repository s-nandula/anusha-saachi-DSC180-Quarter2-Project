# Enhancing Historical Understanding with Retrieval Augmented Generation

This Quarter 2 Project attempts to address historical information retrieval by integrating Retrieval Augmented Generation (RAG) and Large Language Models (LLMs) with Semantic Search. It aims to overcome the limitations of conventional search methods by searching from a range of historical news sources to ensure the credibility and relevance of the information provided.


# Objectives
- Improving Information Access: Enhance accessibility to accurate historical data through innovative search methodologies.
- Addressing Search Engine Limitations: Offer a reliable source of historical information, circumventing the limitations of conventional search engines.
- Providing an Educational Tool: Support educational and research initiatives by delivering comprehensive historical context and reliable sources.
- Catering to Specific Historical Inquiries: Provide targeted responses to users seeking precise historical information.
- Ensuring Data Integrity and Transparency: Maintain high standards of data integrity and transparency throughout the project.

# Dataset
Our curated dataset consists of newspaper data from ProQuest TDM, filtered by articles only, spanning the years 1925 to 1966. It includes publications like the Chicago Tribune, Los Angeles Times, New York Times, Wall Street Journal, and Washington Post, focusing on optimizing space and memory costs by excluding non-article items.


# Project Structure
The repository is organized as follows:
```
anusha-saachi-DSC180-Quarter2-Project/
├─ Current Workflow
├─ ├─ Create_Chroma_DB.ipynb
├─ ├─ RAG_Pipeline (1).ipynb
├─ ├─ SQLite-Database-Creation.ipynb
├─ Previous Iterations
├─ ├─ Data_Cleaning_Trial_Dummy_Data.ipynb
├─ ├─ Llamas-test-Notebook.ipynb
├─ ├─  Previous_Methodology_Create_Embeddings.ipynb
├─ ├─ Previous_Methodology_Search.ipynb
├─ website
├─ ├─ index.html
├─ ├─ style.css
├─ ├─ script.js
├─ README.md
├─ Run_This.ipynb
├─ constants.py
├─ main.py
├─ utils.py

```

# Usage

## Run Locally: 
Please Note: This issue was discussed with both our mentor and TA, and both were made aware that the reproducibility of this code could be affected by the protected environment.

If you wish to run this locally, it will not run unless you have your own SQLite database/ documents. This code is specifically designed to work on the non-exportable data in TDM studios. Also, the dependencies and structure of the code are heavily influenced by the limitations and accessible tools of the TDM studio virtual environment.

1. Clone this repository on your local machine
2. Open your terminal
3. Change (cd) into the directory to the cloned repository
4. Make sure all the necessary packages for running the project all installed
5. Use End-to-End--Database-Creation.py to create your sqlite table
6. Use search.py to execute the search algorithm; replace "query" with the query you would like to run the search algorithm on.

# Website

https://saachishenoy.github.io/Capstone-Website/index.html

## Requirements
1) Python 3
2) Libraries listed in search.py and End-to-End--Database-Creation.py

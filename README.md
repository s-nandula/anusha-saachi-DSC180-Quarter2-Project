# Enhancing Historical Understanding with Retrieval Augmented Generation

This Quarter 2 Project attempts to address historical information retrieval by integrating Retrieval Augmented Generation (RAG) and Large Language Models (LLMs) with Semantic Search. It aims to overcome the limitations of conventional search methods by searching from a range of historical news sources to ensure the credibility and relevance of the information provided.

# Overview

The "Enhancing Historical Understanding with Retrieval Augmented Generation" project is a pioneering effort to bridge the gap between the vast amount of historical data available online and the need for accurate, accessible historical information. In the face of challenges such as the overwhelming volume of resources and the difficulty in discerning credible sources, our project employs advanced technological solutions to sift through historical newspaper archives. By doing so, it aims to provide users with reliable, context-rich insights into history that enhance educational and research endeavors.

At the core of our project is the innovative use of historical newspaper data, spanning from 1925 to 1966, which includes prestigious publications like the Chicago Tribune, Los Angeles Times, New York Times, Wall Street Journal, and Washington Post. Our methodology revolves around the creation of a curated dataset, the development of efficient data parsing and retrieval mechanisms, and the application of advanced machine learning techniques such as document encoding and similarity search. These processes enable us to offer a nuanced approach to historical information retrieval that addresses the limitations of conventional search engines and supports specific historical inquiries with high integrity and transparency.

Through the integration of Retrieval Augmented Generation (RAG) techniques, our project not only improves access to historical data but also contributes to the academic community by fostering collaboration and engagement. It stands as a testament to the potential of data science and artificial intelligence in enhancing our understanding of history, with implications that extend beyond academia to include educational and professional settings where access to reliable historical information is crucial.


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

# Usage
To use the question-answering system:

1. Navigate to the ProQuest TDM Studios at https://tdmstudio.proquest.com/home 
2. Sign in with the following credential
    Email: tdmstudio@clarivate.com
    PW:  UCSDproject2024
4. Head to Workbench #1230
5. Run the Run_This.ipynb notebook (note the first search after starting the kernel may be slow, but it should speed up subsequently)
6. When prompted, enter your query and wait for the system to process and return an answer.

# Files Description
constants.py: Contains the configuration constants for the model, tokenizer, and persistence directory paths.
utils.py: Provides utility functions for initializing embeddings, vector database, language model, and the QA pipeline.
main.py: The main script that brings together all components to process user queries and generate answers.
Run_This.ipynb: A Jupyter notebook designed to demonstrate the project's functionality in an interactive environment.

# Website

https://saachishenoy.github.io/Capstone-Website/index.html

## Requirements
1) Python 3
2) Libraries listed in search.py and End-to-End--Database-Creation.py

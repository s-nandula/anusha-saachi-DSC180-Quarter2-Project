# Enhancing Historical Understanding with Retrieval Augmented Generation

# Overview

Although the amount of resources available on the internet seems to be growing daily, it’s becoming increasingly challenging to navigate the sea of information. While one might expect the influx of articles to make it easier to answer questions, the information accessible is often written well after the historical events have concluded, reflecting modern perspectives and interpretations. When answering historical questions, people often tend to rely on a simple Google search, which leads to websites such as Wikipedia. These sites can be problematic due to the unreliability of sources and the tendency to project 21st-century viewpoints onto historical information. These tools provide users with a surface-level summary that lacks the historical nuance and contextualization that is needed for a thorough understanding of the matter at hand. This project aims to address this gap in relevant and credible information retrieval. This project accomplishes these goals through Retrieval Augmented Generation (RAG), which is a combination of a search algorithm and Large Language Models (LLMs) to answer user queries. A user would be able to ask a question about historical events, and our model would search, retrieve, and synthesize data, ultimately responding to the user’s questions using a diverse array of historical news sources. 


# Objectives
- Addressing Search Engine Limitations:Alleviate the limitations of conventional search engines by offering users a curated database of historical information.
- Providing an Educational Tool:Support educational and research initiatives by providing comprehensive historical context.
- Addressing Specific Historical Inquiries:Cater to the needs of users seeking precise historical information by offering targeted responses to their inquiries.


# Dataset
Our curated dataset consists of newspaper data from ProQuest TDM, filtered by articles only, spanning the years 1925 to 1929. It includes publications like the Chicago Tribune, Los Angeles Times, New York Times, Wall Street Journal, and Washington Post, focusing on optimizing space and memory costs by excluding non-article items.


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
5. Run the ```Run_This.ipynb``` notebook (note: the first search after starting the kernel may be slow, but it should speed up subsequently)
6. When prompted, enter your query and wait for the system to process and return an answer.

# Files Description
- constants.py: Contains the configuration constants for the model, tokenizer, and persistence directory paths.
- utils.py: Provides utility functions for initializing embeddings, vector database, language model, and the QA pipeline.
- main.py: The main script that brings together all components to process user queries and generate answers.
- Run_This.ipynb: A Jupyter notebook designed to demonstrate the project's functionality in an interactive environment.
- README.md: A Markdown file providing an overview of the project, including its purpose, how to set it up, and how to use it.
- website: A folder containing the files/code used to set up the website.
- Previous Iterations: A folder containing the previous methodology, notebooks used to test technologies, and former versions of the tool.
    - Data_Cleaning_Trial_Dummy_Data.ipynb: A  notebook that  focuses on experimenting with data-cleaning techniques using dummy data.
    - Llamas-test-Notebook.ipynb: This notebook is a test of the 'llamas' LLM to experiment with the package.
    - Previous_Methodology_Create_Embeddings.ipynb: A notebook detailing a previous methodology for creating embeddings.
    - Previous_Methodology_Search.ipynb: This notebook demonstrates the former end-to-end methodology using previously created embeddings.
 - Current Workflow: The files outline/display the current methodology/workflow.
    - Create_Chroma_DB.ipynb: A Jupyter notebook used for creating a chroma vector database from article files.
    - RAG_Pipeline (1).ipynb: A notebook implementing the Retrieval-Augmented Generation (RAG) Pipeline for generating answers by combining a retrieval search with a generative model.
    - SQLite-Database-Creation.ipynb: This notebook is responsible for the creation and initialization of an SQLite database for articles and metadata.


# Website

https://saachishenoy.github.io/Capstone-Website/index.html


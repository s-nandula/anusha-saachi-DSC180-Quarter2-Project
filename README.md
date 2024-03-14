# Enhancing Historical Understanding with Retrieval Augmented Generation

# Overview

Although the amount of resources available on the internet seems to be growing daily, it’s becoming increasingly challenging to navigate the sea of information. While one might expect the influx of articles to make it easier to answer questions, the information accessible is often written well after the historical events have concluded, reflecting modern perspectives and interpretations. When answering historical questions, people often tend to rely on a simple Google search, which leads to websites such as Wikipedia. These sites can be problematic due to the unreliability of sources and the tendency to project 21st-century viewpoints onto historical information. These tools provide users with a surface-level summary that lacks the historical nuance and contextualization that is needed for a thorough understanding of the matter at hand. This project aims to address this gap in relevant and credible information retrieval. This project accomplishes these goals through Retrieval Augmented Generation (RAG), which is a combination of a search algorithm and Large Language Models (LLMs) to answer user queries. A user would be able to ask a question about historical events, and our model would search, retrieve, and synthesize data, ultimately responding to the user’s questions using a diverse array of historical news sources. 


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

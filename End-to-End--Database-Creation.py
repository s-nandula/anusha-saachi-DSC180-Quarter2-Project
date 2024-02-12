#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import os
import langchain
import lxml
from lxml import etree
from bs4 import BeautifulSoup
import sqlite3
import xml.etree.ElementTree as ET
from gensim.utils import simple_preprocess
from gensim.corpora.dictionary import Dictionary
from gensim.models import TfidfModel
from gensim import similarities


# In[9]:


data = 'data/All_Publications/'


# In[10]:


input_files = os.listdir(data)


# In[11]:


# Function to strip html tags from text portion
def strip_html_tags(text):
    stripped = BeautifulSoup(text).get_text().replace('\n', ' ').replace('\\', '').strip()
    return stripped


# In[12]:


conn = sqlite3.connect('subset_data.db')
cursor = conn.cursor()


# In[13]:


cursor.execute('''
    CREATE TABLE IF NOT EXISTS subset_table (
        goid INTEGER PRIMARY KEY,
        title TEXT,
        date TEXT,
        publication TEXT,
        text TEXT
    )
''')
conn.commit()


# In[15]:


def insert_data_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Example: Extracting values with handling for missing elements or attributes
    goid = root.find('.//GOID').text if root.find('.//GOID') is not None else None
    title = root.find('.//Title').text if root.find('.//Title') is not None else None
    date = root.find('.//NumericDate').text if root.find('.//NumericDate') is not None else None
    publication = root.find('.//PublisherName').text if root.find('.//PublisherName') is not None else None
#     text = root.find('.//FullText').text if root.find('.//FullText') is not None else None
    if root.find('.//FullText') is not None:
        text = root.find('.//FullText').text

    elif root.find('.//HiddenText') is not None:
        text = root.find('.//HiddenText').text

    elif root.find('.//Text') is not None:
        text = root.find('.//Text').text

    else:
        text = None

    if text is not None:
        text = strip_html_tags(text)
    

    # Use the extracted values as needed (e.g., insert into a database)
#     print("Text:", text)
    # Extract other fields as needed

    # Insert data into SQLite
    cursor.execute('''
        INSERT INTO subset_table (goid, title, date, publication, text)
        VALUES (?, ?, ?, ?, ?)
    ''', (goid, title, date, publication, text))
    # Adjust the query and parameters based on your table schema

# Specify the directory containing your XML files
xml_directory = data

# Iterate through XML files in the directory and insert data into SQLite
for filename in os.listdir(xml_directory):
    if filename.endswith('.xml'):
        xml_file_path = os.path.join(xml_directory, filename)
        insert_data_from_xml(xml_file_path)

# Commit changes and close the database connection
conn.commit()
conn.close()


# In[ ]:





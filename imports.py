from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.indexes import SQLRecordManager, index as LangChainIndex
from langchain_experimental.text_splitter import SemanticChunker
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
import copy
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    PyPDFLoader,
    Docx2txtLoader,
    JSONLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader,
    UnstructuredExcelLoader,
    UnstructuredXMLLoader,
    WebBaseLoader,
    YoutubeLoader,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.sitemap import SitemapLoader
import nest_asyncio
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import scrapetube
import re
import os
from langchain_community.document_loaders import TextLoader
import tiktoken
#from ragas.testset.generator import TestsetGenerator
# from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pandas as pd
import numpy as np
import openai
import os
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import google.generativeai as genai
from langchain_openai import ChatOpenAI
import json
import io
import contextlib
from langchain_google_genai import ChatGoogleGenerativeAI
from datasets import Dataset
from ragas.metrics.critique import harmfulness, maliciousness, coherence, correctness, conciseness
from ragas.metrics import answer_relevancy, answer_similarity, answer_correctness
from ragas import evaluate
import textstat
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import math
import random

nltk.download('punkt')
import Levenshtein
from openai import OpenAI
from dotenv import load_dotenv
import multiprocessing

load_dotenv()  # take environment variables from .env.
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

embeddings = OpenAIEmbeddings()

chroma_db = Chroma(persist_directory=f"./VectorStorage",
                   embedding_function=OpenAIEmbeddings())
MinChunkSize = 200
MaxChunkSize = 1000
ChunkOverLap = 150
AverageChunkSize = 500
namespace = f"VectorStorage/chroma"

record_manager = SQLRecordManager(namespace,
                                  db_url="sqlite:///record_manager_cache.sql")
record_manager.create_schema()

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
patternforChannel = r"/channel/([\w-]+)"
patternforPlayList = r"&list=([A-Za-z0-9_-]+)"
prefix = "https://www.youtube.com/watch?v="

generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
critic_llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()
# #generator = TestsetGenerator.from_langchain(generator_llm, critic_llm,
#                                             embeddings)

safety_settings_NONE = [{
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
}, {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
}, {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
}, {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
}]

llm = ChatGoogleGenerativeAI(model="gemini-pro")
llm.client = genai.GenerativeModel(model_name='gemini-pro',
                                   safety_settings=safety_settings_NONE)
llm = ChatOpenAI(model="gpt-4o")
client = OpenAI()

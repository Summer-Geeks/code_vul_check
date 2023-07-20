import streamlit as st
from streamlit.components.v1 import html
import base64
import requests
import re
from pytube import YouTube
import os
import stat
import subprocess
import random
import urllib.request
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import spacy
import requests
import numpy as np
import pandas as pd
import shutil
from pyppeteer import launch
from generate_mermaid_svg import generate_mermaid_svg

from langchain.document_loaders import WebBaseLoader, GitLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import AgentOutputParser
from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from typing import List, Union, Optional, Type
from pydantic import root_validator, BaseModel, Field, ValidationError, validator
import openai
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List

import base64
import tempfile
import re
import time
import os
import subprocess
import json

OPENAI_API_KEY = ["LIST_OF_YOUR_OPENAI_API_KEYS"]
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
GOOGLE_CX = "YOUR_SEARCH_ENGINE_CX"
os.environ["SERPER_API_KEY"] = "YOUR_SERPER_API_KEY"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_HUGGINGFACE_KEY"

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download("punkt")
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from langchain import HuggingFaceHub
llm = OpenAI(openai_api_key=OPENAI_API_KEY[random.randint(0, len(OPENAI_API_KEY)-1)])
nlp = spacy.load("en_core_web_md")

# Set custom Streamlit theme
st.set_page_config(
    page_title="Chat with AI Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

def onerror(func, path, exc_info):
    if isinstance(exc_info[1], FileNotFoundError):
        return
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

class CodeStore:

    def __init__(
            _self,
            openai_api_key: str,
            github_url: str,
            local_path: str,
            branch: str
    ):
        _self.github_url = github_url
        _self.local_path = local_path
        _self.branch = branch
        _self.OPENAI_API_KEY = openai_api_key
        _self.documents = None
        last_slash_index = github_url.rfind("/")
        last_suffix_index = github_url.rfind(".git")
        if last_suffix_index < 0:
            last_suffix_index = len(github_url)
        _self.persist_directory = github_url[last_slash_index+1 : last_suffix_index]
        _self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _self.code_store = None
        _self.question_answer = None
        _self.chat_history = []

    # @st.cache_data
    def generate_documents(_self):
        _self.clear_files()
        loader = GitLoader(
            clone_url = _self.github_url,
            repo_path = _self.local_path,
            branch = _self.branch
        )

        docs = loader.load()

        text_splitter = CharacterTextSplitter(
            chunk_size = 1500,
            chunk_overlap = 0
        )

        _self.documents = text_splitter.split_documents(docs)
        print(_self.documents)
        # documents are generated

    # @st.cache_data
    def create_vector_store(_self):
        _self.code_store = Chroma.from_documents(
            documents=_self.documents,
            embedding=_self.embedding_function,
            persist_directory=_self.persist_directory,
        )
        prompt = PromptTemplate.from_template(
                """
                Question: {question}
                Instructions: \n
                    Respond to the human as helpfully and accurately as possible in 100 words or less. You can look into the github code repository, YOU HAVE BEEN GIVEN ACCESS TO ALL FILES.

                    Output Format must be:
                        Brief Explanation : <Give Brief Explanation here> \n
                        Code: <give the relevant code here>\n

                Example:
                    OUTPUT:
                        BRIEF EXPLANATION: This folder contains ...
                        Code:
                            
                            def function_Get():
                                pass
                            

                    OUTPUT:
                        BRIEF EXPLANATION: The function present in the code ...
                        Code:
                            
                            def myfunction():
                                pass
                            

                    OUTPUT:
                        BRIEF EXPLANATION: This Algorithm is used to ...
                        Code:
                            
                            def algorithm():
                                pass
                            
                """
            )

        # prompt = PromptTemplate.from_template(
        #     "Question: {question}"
        #     "Instructions: \n"
        #     "Output Format must be:"
        #     "Explanation : <Give Explanation here> \n"
        #     "Code: <give the relevant code here>\n"
        #     "Summary: <give summary here>"
        # )

        if _self.code_store:
            print("Code Store exists")

        _self.code_store.persist()

        _self.question_answer = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(
                openai_api_key = _self.OPENAI_API_KEY,
                temperature = 0
            ),
            _self.code_store.as_retriever(),
            return_source_documents = True,
            condense_question_prompt=prompt
            # get_chat_history = _self.get_chat_history
        )
        
    @st.cache_data
    def search_code(_self, query: str):
        # Preprocess and summarize the input query
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(query)
        filtered_query = [w for w in word_tokens if not w in stop_words]
        summarized_query = ' '.join(filtered_query)

        # Pass the summarized query to the question_answer method
        _self.chat_history = []
        result = _self.question_answer({
            "question": summarized_query,
            "chat_history": _self.chat_history
        })
        _self.chat_history.append((summarized_query, result["answer"]))

        return result

    def clear_files(_self):
        try:
            # subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", _self.local_path], check=True)
            # subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", _self.persist_directory], check=True)
            shutil.rmtree(_self.local_path, onerror=onerror)
            shutil.rmtree(_self.persist_directory, onerror=onerror)

            return True
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")
        # except subprocess.CalledProcessError as e:
        #     print(f"Failed to remove Code Store or Cloned Repository: {e}")
        #     return False

    def clear_history(_self):
        _self.chat_history = []

    def reinitialize(_self, openai_api_key: str, github_url: str, local_path: str, branch: str):
        _self.github_url = github_url
        _self.local_path = local_path
        _self.branch = branch
        _self.OPENAI_API_KEY = openai_api_key
        _self.documents = None
        last_slash_index = github_url.rfind("/")
        last_suffix_index = github_url.rfind(".git")
        if last_suffix_index < 0:
            last_suffix_index = len(github_url)
        _self.persist_directory = github_url[last_slash_index+1 : last_suffix_index]
        _self.embedding_function = OpenAIEmbeddings(openai_api_key = openai_api_key)
        _self.code_store = None
        _self.question_answer = None
        _self.chat_history = []

@st.cache_data
def _cr(github_url, git_branch):
    cr = CodeStore(
                openai_api_key=OPENAI_API_KEY[random.randint(0, len(OPENAI_API_KEY)-1)],
                github_url=github_url,
                local_path="./repo",
                branch=git_branch)
    return cr

class UseCaseGenerator:
    def __init__(_self, api_key, description, diagram):
        openai.api_key = api_key
        _self.description = description
        _self.diagram = diagram

    # @st.cache_data
    def generate_use_cases(_self, _description, diagram):
        prompt = f"""Generate structured and well-defined use cases for the given system description, and then generate mermaid code for the given diagram type.
        System Description: {_description}
        Use Case 1:
        Use Case 2:
        Use Case 3:
        Use Case n:

        Diagram Type: {diagram}

        Mermaid Code:
        """

        completions = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
        )

        message = completions.choices[0].text.strip()
        return message

class VulnerabilityStore:

    def __init__(_self, openai_api_key:list):

        _self.OPENAI_API_KEY = openai_api_key
        _self.documents = []
        _self.summarizer_llm = HuggingFaceHub(
            repo_id="sshleifer/distilbart-cnn-12-6", model_kwargs={"temperature": 0.2, "max_length": 10000}
        )
        _self.similarity_model = SentenceTransformer('paraphrase-distilroberta-base-v2')
        _self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _self.start_urls = [
            "https://owasp.org/Top10/A01_2021-Broken_Access_Control",
            "https://owasp.org/Top10/A02_2021-Cryptographic_Failures",
            "https://owasp.org/Top10/A03_2021-Injection",
            "https://owasp.org/Top10/A04_2021-Insecure_Design",
            "https://owasp.org/Top10/A05_2021-Security_Misconfiguration",
            "https://owasp.org/Top10/A06_2021-Vulnerable_and_Outdated_Components",
            "https://owasp.org/Top10/A07_2021-Identification_and_Authentication_Failures",
            "https://owasp.org/Top10/A08_2021-Software_and_Data_Integrity_Failures",
            "https://owasp.org/Top10/A09_2021-Security_Logging_and_Monitoring_Failures",
            "https://owasp.org/Top10/A10_2021-Server-Side_Request_Forgery_%28SSRF%29"
        ]
        _self.labels = []
        _self.persist_directories = []
        _self.metadata = []
        _self.stores: list[Chroma] = []
        _self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.4; Win64; x64) Gecko/20130401 Firefox/60.7",
            "Mozilla/5.0 (Linux; Linux i676 ; en-US) AppleWebKit/602.37 (KHTML, like Gecko) Chrome/53.0.3188.182 Safari/533",
            "Mozilla/5.0 (Android; Android 6.0.1; SM-G928S Build/MDB08I) AppleWebKit/536.43 (KHTML, like Gecko)  Chrome/53.0.1961.231 Mobile Safari/602.4",
            "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_1_7) Gecko/20100101 Firefox/48.4",
            "Mozilla/5.0 (Windows NT 10.1; Win64; x64) AppleWebKit/602.6 (KHTML, like Gecko) Chrome/49.0.2764.174 Safari/603",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 9_8_7) Gecko/20130401 Firefox/59.7",
            "Mozilla/5.0 (Linux; U; Android 5.0; LG-D332 Build/LRX22G) AppleWebKit/602.2 (KHTML, like Gecko)  Chrome/54.0.2014.306 Mobile Safari/600.5",
            "Mozilla/5.0 (Android; Android 5.0.2; HTC 80:number1-2w Build/LRX22G) AppleWebKit/600.31 (KHTML, like Gecko)  Chrome/54.0.3164.157 Mobile Safari/603.5",
            "Mozilla/5.0 (Linux; Android 7.1; Nexus 8P Build/NPD90G) AppleWebKit/601.15 (KHTML, like Gecko)  Chrome/47.0.3854.218 Mobile Safari/600.6",
            "Mozilla/5.0 (Android; Android 5.1; SM-G928L Build/LRX22G) AppleWebKit/602.47 (KHTML, like Gecko)  Chrome/48.0.1094.253 Mobile Safari/537.6",
        ]
        _self.question_answer = []

    def get_links_from_url(_self, url: str):
        req = urllib.request.Request(url, headers={
            "User-Agent": _self.user_agents[random.randint(0, 9)]
        })
        html_page = urllib.request.urlopen(req)
        soup = BeautifulSoup(html_page, 'html.parser')
        links = []
        for link in soup.findAll('a'):
            links.append(urljoin(url, link.get('href')))
        return links
    
    @st.cache_data
    def expand_start_urls(_self):
        to = len(_self.start_urls)
        for l in range(to):
            try:
                if len(_self.start_urls) > 0:
                    break
                newlinks = _self.get_links_from_url(url = _self.start_urls[l])
                checked_links = []
                for url in newlinks:
                    if len(url) < 10:
                        continue
                    if url[-1] == "/":
                        url = url[:-1]
                    checked_links.append(url)
                checked_links = list(set(checked_links))
                # for i in range(len(checked_links)):
                    # _self.labels.append("vulnerability_related")
                _self.start_urls.extend(checked_links)
                # if len(_self.labels) != len(_self.start_urls):
                #     print("Gadbad :)")
            except Exception:
                # print("URL : ", _self.start_urls[l], "Error: ", e)
                continue

    @st.cache_data
    def get_documents(_self):
        for url in _self.start_urls:
            try:
                loader = WebBaseLoader(web_path=url)
                loader.default_parser = "html.parser"
                doc = loader.load()

                parser = HtmlParser.from_url(url, Tokenizer("english"))
                summarizer = LuhnSummarizer()
                summary = str(summarizer(parser.document, sentences_count=4))
                summary = summary.replace("<", "").replace(">", "").replace("Sentence:", "")
                for d in doc:
                    if d.metadata is None:
                        d.metadata = {}
                    d.metadata["summary"] = summary.replace("<", "").replace(">", "").replace("Sentence:", "")
                _self.documents.append(doc)
                _self.metadata.append(summary)

            except Exception as e:
                print("URL : ", url, "Error : ", e)
                continue

    @staticmethod
    def clear_vector_stores():
        try:
            subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", "vectorstores"])
            print("Removed vectorstores")
        except subprocess.CalledProcessError:
            print("Failure")

    @st.cache_data
    def create_vector_stores(_self):
        # _self.clear_vector_stores()
        l = len(_self.documents)
        p = 0
        for i in range(l):
            try:
                _self.stores.append(
                    Chroma.from_documents(
                        documents=_self.documents[i],
                        embedding=_self.embedding_function,
                        persist_directory=_self.persist_directories[i]
                    )
                )
            except Exception as e:
                print(e, _self.documents[i])
                p = p+1
                continue

            _self.stores[i-p].persist()
            prompt = PromptTemplate.from_template(
            "Question: {question}"
            "Instructions: \n"
            "Output Format must be:"
            "Explanation : <Give Explanation here> \n"
            "Code: <give the relevant code here>\n"
            "Summary: <give summary here>"
            )
            # make qa
            _self.question_answer.append(ConversationalRetrievalChain.from_llm(
                ChatOpenAI(
                    openai_api_key = _self.OPENAI_API_KEY[random.randint(0,len(_self.OPENAI_API_KEY)-1)],
                    temperature = 0
                ),
                retriever=(_self.stores[i-p]).as_retriever(search_kwargs={"k": 1}), condense_question_prompt = prompt
            ))
    @st.cache_data
    def add_url(_self, url: str, label: str):
        loader = WebBaseLoader(web_path=url)
        loader.default_parser = "html.parser"
        doc = loader.load()

        if len(doc) == 0:
            return

        parser = HtmlParser.from_url(url, Tokenizer("english"))

        ## LUHN method
        summarizer = LuhnSummarizer()
        summary = str(summarizer(parser.document, sentences_count=4))
        summary = summary.replace("<", "").replace(">", "").replace("Sentence:", "")

        for d in doc:
            try:
                if d.metadata is None or d.metadata == {}:
                    d.metadata = {"error": "metadata was null"}
                d.metadata["summary"] = summary.replace("<", "").replace(">", "").replace("Sentence:", "")
                d.metadata["url"] = url
            except Exception:
                pass
                # print(e)
                # print(d)

        topic_exists = False
        for i in range(len(_self.metadata)):
            qe = _self.similarity_model.encode(doc[0].metadata["summary"])
            me = _self.similarity_model.encode(_self.metadata[i])
            score = qe.dot(me) / (np.linalg.norm(qe) * np.linalg.norm(me))
            if score > 0.6:
                topic_exists = True
                print("merged")
                try:
                    for d in doc:
                        _self.stores[i].add_documents(d)
                    _self.metadata[i] += " " + summary
                except Exception:
                    pass
                    # print(score, i, _self.persist_directories[i])
                break

        if not topic_exists:
            per = ""
            if url[-1] == "/":
                url = url[:-1]
            slash = url.rfind("/")
            if slash < 0:
                ext = url.rfind(".")
                per = "./vectorstores/" + url[ext+1:]
            if slash > 0:
                per = "./vectorstores/" + url[slash+1:]
            _self.stores.append(
                Chroma.from_documents(
                    documents=doc,
                    embedding=_self.embedding_function,
                    persist_directory=per
                )
            )
            _self.labels.append(label)
            _self.stores[-1].persist()
            _self.documents.append(doc)
            _self.metadata.append(summary)
            _self.persist_directories.append(per)

            prompt = PromptTemplate.from_template(
            "Question: {question}"
            "Instructions: \n"
            "Output Format must be:"
            "Explanation : <Give Explanation here> \n"
            "Code: <give the relevant code here>\n"
            "Summary: <give summary here>"
            )

                        # make qa
            _self.question_answer.append(ConversationalRetrievalChain.from_llm(
                llm = ChatOpenAI(temperature = 0.2, openai_api_key = _self.OPENAI_API_KEY[random.randint(0,len(_self.OPENAI_API_KEY)-1)]),
                retriever=_self.stores[-1].as_retriever(search_kwargs={"k": 1}),
                max_tokens_limit=4000, condense_question_prompt = prompt
            ))

    @st.cache_data
    def run_vul(_self):
        for url in _self.start_urls:
            try:
                _self.add_url(url = url, label="vulnerability_related")
            except Exception:
                # print("Error in URL : ", url)
                # print("Skip due to error : ", e)
                continue

            # print(len(_self.metadata), len(_self.documents), len(_self.stores), len(_self.persist_directories), len(_self.question_answer))
            print("URL : ", url)
            print("Store : ", _self.persist_directories[-1])
            # print("Metadata : ", _self.metadata[-1])

    @st.cache_data
    def search(_self, query:str, label:str):
        similarity_scores = []
        l = len(_self.metadata)
        for i in range(l):
            # if store or metadata is not labelled related to the label mentioned then skip
            if _self.labels[i] != label:
                similarity_scores.append(0)
                continue

            qe = _self.similarity_model.encode(_self.metadata[i])
            me = _self.similarity_model.encode(query)

            score = qe.dot(me) / (np.linalg.norm(qe) * np.linalg.norm(me))
            similarity_scores.append(score)

        solutions = []
        for i in range(l):
            if similarity_scores[i] > 0.5:
                solutions.append(i)

        outputs = []
        chat_history = []
        if len(solutions) == 0:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": "YOUR_GOOGLE_API_KEY",
                "cx": "YOUR_SEARCH_ENGINE_CX_KEY",
                "q": query,
                "num": 10,
                "fields": "items(link)"
            }
            response = requests.get(url, params=params)

            if response.status_code == 200:
                results = response.json()["items"]
                urls = [result["link"] for result in results]
                text_urls = []
                for url in urls:
                    try:
                        response = requests.get(url)
                        if response.status_code == 200 and 'text' in response.headers['Content-Type']:
                            text_urls.append(url)
                        elif response.status_code == 200 and 'text' in response.content.decode('utf-8'):
                            text_urls.append(url)
                    except:
                        pass
                stprev = len(_self.stores)
                print(text_urls)
                for i in text_urls:
                    try:
                        _self.add_url(i, label)
                        _self.start_urls.append(i)
                    except Exception as e:
                        print("URL", i, e)
                stnext = len(_self.stores)

                #  label new stores generated with the label provided
                # for i in range(stnext - stprev):
                #     _self.labels.append(label)

                try:
                    l = len(_self.question_answer)
                    for i in range(l- (stnext - stprev), l):
                        qa = _self.question_answer[i]
                        result = qa({"question": query, "chat_history": []})
                        outputs.append(result)
                except Exception:
                    pass
                    # print("Gadbad", e, "\n", "LEngth : ", len(_self.start_urls), "QA len : ", len(_self.question_answer), " sol len ", len(solutions))

            else:
                outputs.append("No search results found.")

        else:
            try:
                for i in solutions:
                    qa = _self.question_answer[i]
                    result = qa({"question": query, "chat_history": []})
                    outputs.append(result)
                    chat_history.append(result)
            except Exception:
                pass
                # print(e)

        return outputs

vul = VulnerabilityStore(openai_api_key=OPENAI_API_KEY)
# vul.clear_vector_stores()
vul.expand_start_urls()
# print(len(vul.start_urls))
vul.run_vul()


class GeneralStore:

    def __init__(self, openai_api_key:list):

        self.OPENAI_API_KEY = openai_api_key
        self.documents = []
        self.summarizer_llm = HuggingFaceHub(
            repo_id="sshleifer/distilbart-cnn-12-6", model_kwargs={"temperature": 0, "max_length": 10000}
        )
        self.similarity_model = SentenceTransformer('paraphrase-distilroberta-base-v2')
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.start_urls = [
            # "https://owasp.org/Top10/A01_2021-Broken_Access_Control",
            # "https://owasp.org/Top10/A02_2021-Cryptographic_Failures",
            # "https://owasp.org/Top10/A03_2021-Injection",
            # "https://owasp.org/Top10/A04_2021-Insecure_Design",
            # "https://owasp.org/Top10/A05_2021-Security_Misconfiguration",
            # "https://owasp.org/Top10/A06_2021-Vulnerable_and_Outdated_Components",
            # "https://owasp.org/Top10/A07_2021-Identification_and_Authentication_Failures",
            # "https://owasp.org/Top10/A08_2021-Software_and_Data_Integrity_Failures",
            # "https://owasp.org/Top10/A09_2021-Security_Logging_and_Monitoring_Failures",
            # "https://owasp.org/Top10/A10_2021-Server-Side_Request_Forgery_%28SSRF%29"
        ]
        self.labels = []
        self.persist_directories = []
        self.metadata = []
        self.stores: list[Chroma] = []
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.4; Win64; x64) Gecko/20130401 Firefox/60.7",
            "Mozilla/5.0 (Linux; Linux i676 ; en-US) AppleWebKit/602.37 (KHTML, like Gecko) Chrome/53.0.3188.182 Safari/533",
            "Mozilla/5.0 (Android; Android 6.0.1; SM-G928S Build/MDB08I) AppleWebKit/536.43 (KHTML, like Gecko)  Chrome/53.0.1961.231 Mobile Safari/602.4",
            "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_1_7) Gecko/20100101 Firefox/48.4",
            "Mozilla/5.0 (Windows NT 10.1; Win64; x64) AppleWebKit/602.6 (KHTML, like Gecko) Chrome/49.0.2764.174 Safari/603",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 9_8_7) Gecko/20130401 Firefox/59.7",
            "Mozilla/5.0 (Linux; U; Android 5.0; LG-D332 Build/LRX22G) AppleWebKit/602.2 (KHTML, like Gecko)  Chrome/54.0.2014.306 Mobile Safari/600.5",
            "Mozilla/5.0 (Android; Android 5.0.2; HTC 80:number1-2w Build/LRX22G) AppleWebKit/600.31 (KHTML, like Gecko)  Chrome/54.0.3164.157 Mobile Safari/603.5",
            "Mozilla/5.0 (Linux; Android 7.1; Nexus 8P Build/NPD90G) AppleWebKit/601.15 (KHTML, like Gecko)  Chrome/47.0.3854.218 Mobile Safari/600.6",
            "Mozilla/5.0 (Android; Android 5.1; SM-G928L Build/LRX22G) AppleWebKit/602.47 (KHTML, like Gecko)  Chrome/48.0.1094.253 Mobile Safari/537.6",
        ]
        self.question_answer = []

    def get_links_from_url(self, url: str):
        req = urllib.request.Request(url, headers={
            "User-Agent": self.user_agents[random.randint(0, 9)]
        })
        html_page = urllib.request.urlopen(req)
        soup = BeautifulSoup(html_page, 'html.parser')
        links = []
        for link in soup.findAll('a'):
            links.append(urljoin(url, link.get('href')))
        return links

    def expand_start_urls(self):
        to = len(self.start_urls)
        for l in range(to):
            try:
                if len(self.start_urls) > 0:
                    break
                newlinks = self.get_links_from_url(url = self.start_urls[l])
                checked_links = []
                for url in newlinks:
                    if len(url) < 10:
                        continue
                    if url[-1] == "/":
                        url = url[:-1]
                    checked_links.append(url)
                checked_links = list(set(checked_links))
                # for i in range(len(checked_links)):
                    # self.labels.append("vulnerability_related")
                self.start_urls.extend(checked_links)
                # if len(self.labels) != len(self.start_urls):
                #     print("Gadbad :)")
            except Exception:
                # print("URL : ", self.start_urls[l], "Error: ", e)
                continue
    @st.cache_data
    def get_documents(self):
        for url in self.start_urls:
            try:
                loader = WebBaseLoader(web_path=url)
                loader.default_parser = "html.parser"
                doc = loader.load()

                parser = HtmlParser.from_url(url, Tokenizer("english"))
                summarizer = LuhnSummarizer()
                summary = str(summarizer(parser.document, sentences_count=4))
                summary = summary.replace("<", "").replace(">", "").replace("Sentence:", "")
                for d in doc:
                    if d.metadata is None:
                        d.metadata = {}
                    d.metadata["summary"] = summary.replace("<", "").replace(">", "").replace("Sentence:", "")
                self.documents.append(doc)
                self.metadata.append(summary)

            except Exception as e:
                print("URL : ", url, "Error : ", e)
                continue

    @staticmethod
    def clear_vector_stores():
        try:
            subprocess.run(["rm", "-r", "./generalstore"])
            print("Removed generalstore")
        except subprocess.CalledProcessError:
            print("Failure")

    @st.cache_data
    def create_vector_stores(self):
        # self.clear_vector_stores()
        l = len(self.documents)
        p = 0
        for i in range(l):
            try:
                self.stores.append(
                    Chroma.from_documents(
                        documents=self.documents[i],
                        embedding=self.embedding_function,
                        persist_directory=self.persist_directories[i]
                    )
                )
            except Exception as e:
                print(e, self.documents[i])
                p = p+1
                continue

            self.stores[i-p].persist()
            prompt = PromptTemplate.from_template(
            "Question: {question}"
            "Instructions: \n"
            "Output Format must be:"
            "Explanation : <Give Explanation here> \n"
            "Code: <give the relevant code here>\n"
            "Summary: <give summary here>"
            )
            # make qa
            self.question_answer.append(ConversationalRetrievalChain.from_llm(
                ChatOpenAI(
                    openai_api_key = self.OPENAI_API_KEY[random.randint(0,len(self.OPENAI_API_KEY)-1)],
                    temperature = 0
                ),
                retriever=(self.stores[i-p]).as_retriever(search_kwargs={"k": 1}), condense_question_prompt = prompt
            ))
    @st.cache_data
    def add_url(self, url: str, label: str):
        loader = WebBaseLoader(web_path=url)
        loader.default_parser = "html.parser"
        doc = loader.load()

        if len(doc) == 0:
            return

        parser = HtmlParser.from_url(url, Tokenizer("english"))

        ## LUHN method
        summarizer = LuhnSummarizer()
        summary = str(summarizer(parser.document, sentences_count=4))
        summary = summary.replace("<", "").replace(">", "").replace("Sentence:", "")

        for d in doc:
            try:
                if d.metadata is None or d.metadata == {}:
                    d.metadata = {"error": "metadata was null"}
                d.metadata["summary"] = summary.replace("<", "").replace(">", "").replace("Sentence:", "")
                d.metadata["url"] = url
            except Exception:
                pass
                # print(e)
                # print(d)

        topic_exists = False
        for i in range(len(self.metadata)):
            qe = self.similarity_model.encode(doc[0].metadata["summary"])
            me = self.similarity_model.encode(self.metadata[i])
            score = qe.dot(me) / (np.linalg.norm(qe) * np.linalg.norm(me))
            if score > 0.6:
                topic_exists = True
                print("merged")
                try:
                    for d in doc:
                        self.stores[i].add_documents(d)
                    self.metadata[i] += " " + summary
                except Exception:
                    pass
                    # print(score, i, self.persist_directories[i])
                break

        if not topic_exists:
            per = ""
            if url[-1] == "/":
                url = url[:-1]
            slash = url.rfind("/")
            if slash < 0:
                ext = url.rfind(".")
                per = "./generalstore/" + url[ext+1:]
            if slash > 0:
                per = "./generalstore/" + url[slash+1:]
            self.stores.append(
                Chroma.from_documents(
                    documents=doc,
                    embedding=self.embedding_function,
                    persist_directory=per
                )
            )
            self.labels.append(label)
            self.stores[-1].persist()
            self.documents.append(doc)
            self.metadata.append(summary)
            self.persist_directories.append(per)

            prompt = PromptTemplate.from_template(
            "Question: {question}"
            "Instructions: \n"
            "Output Format must be:\n"
            "Explanation : <Give Explanation here> \n"
            "Code: <give the relevant code here>\n"
            "Summary: <give summary here>\n"
            ""
            )

                        # make qa
            self.question_answer.append(ConversationalRetrievalChain.from_llm(
                llm = ChatOpenAI(temperature = 0, openai_api_key = self.OPENAI_API_KEY[random.randint(0,len(self.OPENAI_API_KEY)-1)]),
                retriever=self.stores[-1].as_retriever(search_kwargs={"k": 1}),
                max_tokens_limit=4000, condense_question_prompt = prompt
            ))


    def run_vul(self):
        for url in self.start_urls:
            try:
                self.add_url(url = url, label="vulnerability_related")
            except Exception:
                # print("Error in URL : ", url)
                # print("Skip due to error : ", e)
                continue

            # print(len(self.metadata), len(self.documents), len(self.stores), len(self.persist_directories), len(self.question_answer))
            print("URL : ", url)
            print("Store : ", self.persist_directories[-1])
            # print("Metadata : ", self.metadata[-1])


    def search(self, query:str, label:str):
        similarity_scores = []
        l = len(self.metadata)
        for i in range(l):
            # if store or metadata is not labelled related to the label mentioned then skip
            if self.labels[i] != label:
                similarity_scores.append(0)
                continue

            qe = self.similarity_model.encode(self.metadata[i])
            me = self.similarity_model.encode(query)

            score = qe.dot(me) / (np.linalg.norm(qe) * np.linalg.norm(me))
            similarity_scores.append(score)

        solutions = []
        for i in range(l):
            if similarity_scores[i] > 0.5:
                solutions.append(i)

        outputs = []
        chat_history = []
        if len(solutions) == 0:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": "YOUR_GOOGLE_API_KEY",
                "cx": "YOUR_SEARCH_ENGINE_CX_KEY",
                "q": query,
                "num": 10,
                "fields": "items(link)"
            }
            response = requests.get(url, params=params)

            if response.status_code == 200:
                results = response.json()["items"]
                urls = [result["link"] for result in results]
                text_urls = []
                for url in urls:
                    try:
                        response = requests.get(url)
                        if response.status_code == 200 and 'text' in response.headers['Content-Type']:
                            text_urls.append(url)
                        elif response.status_code == 200 and 'text' in response.content.decode('utf-8'):
                            text_urls.append(url)
                    except:
                        pass
                stprev = len(self.stores)
                print(text_urls)
                for i in text_urls:
                    try:
                        self.add_url(i, label)
                        self.start_urls.append(i)
                    except Exception as e:
                        print("URL", i, e)
                stnext = len(self.stores)

                #  label new stores generated with the label provided
                # for i in range(stnext - stprev):
                #     self.labels.append(label)

                try:
                    l = len(self.question_answer)
                    for i in range(l- (stnext - stprev), l):
                        qa = self.question_answer[i]
                        result = qa({"question": query, "chat_history": []})
                        outputs.append(result)
                except Exception:
                    pass

            else:
                outputs.append("No search results found.")

        else:
            try:
                for i in solutions:
                    qa = self.question_answer[i]
                    result = qa({"question": query, "chat_history": []})
                    outputs.append(result)
                    chat_history.append(result)
            except Exception:
                pass

        return outputs
    
gs = GeneralStore(openai_api_key=OPENAI_API_KEY)


"""# Classifier"""
dataset = None
if dataset not in st.session_state:
    dataset = pd.read_csv("code_chk_data.csv", names=["Query", "Type"])
    print(dataset)

class ClassifierModel:
    def __init__(_self, data, criteria, model_name="all-MiniLM-L6-v2"):
        _self.scaler = None
        _self.embeddings = None
        _self.classifier = None
        _self.Y_test = None
        _self.Y_train = None
        _self.X_test = None
        _self.X_train = None
        _self.Y = None
        _self.X = None
        _self.punctuations = None
        _self.nlp = spacy.load("en_core_web_sm")
        _self.stop_words = _self.nlp.Defaults.stop_words
        # filtered_dataset = data.loc[(data["Type"] == criteria[0]) | (data["Type"] == criteria[1])]
        # _self.dataset = filtered_dataset
        _self.dataset = data
        _self.criteria = criteria
        _self.model = SentenceTransformer(model_name)

    def split(_self):
        if len(_self.X) != len(_self.Y):
            print("Length of X not equal to Y", len(_self.X), len(_self.Y), len(dataset), len(_self.embeddings))
            return
        _self.X_train, _self.X_test, _self.Y_train, _self.Y_test = train_test_split(_self.X, _self.Y, test_size=0.20)

    def tokenizer(_self, sentence):
        words = _self.nlp(sentence)
        _self.punctuations = "!#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
        tokens = [word.lemma_.lower().strip() for word in words]
        tokens = [word for word in tokens if word not in _self.stop_words and word not in _self.punctuations]
        sentence = " ".join(tokens)
        return sentence

    def tokenize(_self):
        _self.dataset["Tokenized"] = _self.dataset["Query"].apply(_self.tokenizer)

    def encoder(_self):
        _self.embeddings = _self.dataset["Tokenized"].apply(_self.model.encode)
        _self.X = _self.embeddings.to_list()
        _self.scaler = LabelEncoder().fit(list(dataset["Type"]))
        _self.Y = _self.scaler.transform(list(dataset["Type"]))

    @st.cache_data
    def train_classifier(_self):
        _self.classifier = SVC(kernel="rbf", random_state=0)
        _self.classifier.fit(_self.X_train, _self.Y_train)

    def check_test(_self):
        Y_predict = _self.classifier.predict(_self.X_test)
        cm = confusion_matrix(_self.Y_test, Y_predict)
        print("Confusion matrix", cm)
        print("Accuracy Score : ", accuracy_score(_self.Y_test, Y_predict))
        print("Precision Score : ", precision_score(_self.Y_test, Y_predict, average='macro'))
        print("Recall Score : ", recall_score(_self.Y_test, Y_predict, average='macro', zero_division=0))

    def predict_classification(_self, query: str) -> str:
        tokenized_input = _self.tokenizer(sentence=query)
        embedding_input = _self.model.encode(sentences=tokenized_input)
        prediction = _self.classifier.predict([embedding_input])
        # print(_self.scaler.inverse_transform(prediction))
        return _self.scaler.inverse_transform(prediction)[0]

@st.cache_resource
def load_classifier():
    global classifier
    classifier = ClassifierModel(data=st.session_state.dataset, criteria=["code_algo_related", "vulnerability_related", "general_chat"])
    classifier = load_classifier()
    classifier.tokenize()
    classifier.encoder()
    classifier.split()
    classifier.train_classifier()
    classifier.check_test()

"""# Tools"""

# class VulInputSchema(BaseModel):
#     query: str = Field(...)
#     label: str = Field(...)

#     @st.cache_data
#     @root_validator
#     def label_in(cls, values):
#         l = values["label"]
#         # q = values["query"]
#         if l not in ['code_related', 'algo_related', 'vulnerability_related']:
#             raise ValueError(
#                 "Label not in :",
#                 "1. code_related\n 2. algo_related\n3. vulnerability_related"
#             )
#         return values

class vulSearch(BaseTool):
    name: str = "VulSearch"
    description: str = "Use this tool to search for a Code Vulnerability when provided with a query"
    # args_schema: Type[VulInputSchema] = VulInputSchema

    # def _run(self, query:str, label: str):
    def _run(self, query: str):
        time.sleep(60)
        return vul.search(query=query, label="vulnerability_related")

    def _arun(self) -> NotImplementedError:
        return NotImplementedError("This tool does not implement asynchronous ouputs.")
  
class codeVectorStoreTool(BaseTool):
    name = "Code Vector Store"
    description = "Use this tool to view and analyze the code of github repo files and folders."
    @st.cache_data
    def _run(_self, query: str):
        time.sleep(60)
        cr = st.session_state.cr
        return cr.search_code(query)

    def _arun(_self) -> NotImplementedError:
        return NotImplementedError("This tool does not require Async output.")

class generalSearch(BaseTool):
    name: str = "General Chat Search"
    description: str = "Use this tool to search for general information to retrieve from web when provided with a query"
    # args_schema: Type[VulInputSchema] = VulInputSchema

    # def _run(self, query:str, label: str):
    def _run(self, query: str):
        time.sleep(60)
        return gs.search(query=query, label="general_chat")

    def _arun(self) -> NotImplementedError:
        return NotImplementedError("This tool does not implement asynchronous ouputs.")
    

class classifierTool(BaseTool):
    name = "Query Classifier Tool"
    description = "This is the first tool that you must use to know whether the query is related to code or vulnerability or general chat."
    @st.cache_data
    def _run(_self, query: str):
        time.sleep(60)
        return classifier.predict_classification(query=query)
    
    def _arun(_self) -> NotImplementedError:
        return NotImplementedError("This tool does not have async output option.")

# class CVOutputParser(BaseModel):
#     output: str = Field(description="This is the output of the parser it must include explanation with either code or vulnerability")

#     @validator('output')
#     def output_includes(cls, field):
#         if "Explanation:" in field and ("Code:" in field or "Vulnerability:" in field):
#             return field
#         raise ValueError("Output does not include Explanation, Code or Vulnerability.")

# parser = PydanticOutputParser(pydantic_object=CVOutputParser)

# prompt = PromptTemplate(
#     template="Answer the user query.\n{format_instructions}\n{query}\n",
#     input_variables=["query"],
#     partial_variables={"format_instructions": parser.get_format_instructions()},
# )

tools = [classifierTool(), codeVectorStoreTool(), vulSearch(), generalSearch()]

model = ChatOpenAI(
    openai_api_key = OPENAI_API_KEY[random.randint(0, len(OPENAI_API_KEY)-1)],
    # openai_api_key = OPENAI_API_KEY[2],
    temperature = 0
)
from langchain.experimental import load_chat_planner, load_agent_executor, PlanAndExecute

planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)

# agent = initialize_agent(
#     agent = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     tools = tools,
#     llm = ChatOpenAI(
#       openai_api_key = OPENAI_API_KEY[random.randint(0, len(OPENAI_API_KEY)-1)],
#       temperature = 0
#     ),
#     verbose = True,
#     handle_parsing_errors=True,
#     output_parser = parser
# )

# class MyOutputParser(OutputParser):
#     def parse(self, text):
#         # Split the text by the Action and Observation markers
#         parts = text.split("Action:\n")
#         if len(parts) != 2:
#             # No action found, return an AgentFinish with the text as the answer
#             return AgentFinish(answer=text)
#         else:
#             # Extract the action and observation parts
#             action = parts[1].split("Observation:\n")[0].strip()
#             observation = parts[1].split("Observation:\n")[1].strip()
#             # Parse the action as a JSON object
#             action_input = json.loads(action)
#             # Return an AgentAction with the action and action_input
#             return AgentAction(action=action_input["action"], action_input=action_input["action_input"], observation=observation)

# # Create an instance of MyOutputParser
# my_output_parser = MyOutputParser()

# # Create an instance of PlanAndExecute agent with my_output_parser as the output_parser argument
# agent = PlanAndExecute(planner=planner, executor=executor, output_parser=my_output_parser, verbose=True)

class MyOutput(BaseModel):
    action: str
    action_input: dict
    observation: str

    # Add custom parsing logic using a pydantic validator
    @validator('action', pre=True)
    def parse_action(cls, action):
        # Split the text by the Action and Observation markers
        parts = action.split("Action:\n")
        if len(parts) != 2:
            # No action found, return None
            return None
        else:
            # Extract the action and observation parts
            action = parts[1].split("Observation:\n")[0].strip()
            observation = parts[1].split("Observation:\n")[1].strip()
            # Parse the action as a JSON object
            action_input = json.loads(action)
            # Return the parsed action
            return action_input["action"]

    @validator('action_input', pre=True)
    def parse_action_input(cls, action_input, values):
        # Check if the action has been parsed successfully
        if 'action' in values:
            # Split the text by the Action and Observation markers
            parts = action_input.split("Action:\n")
            if len(parts) != 2:
                # No action found, return None
                return None
            else:
                # Extract the action and observation parts
                action = parts[1].split("Observation:\n")[0].strip()
                observation = parts[1].split("Observation:\n")[1].strip()
                # Parse the action as a JSON object
                action_input = json.loads(action)
                # Return the parsed action_input
                return action_input["action_input"]
        else:
            return None

    @validator('observation', pre=True)
    def parse_observation(cls, observation, values):
        # Check if the action has been parsed successfully
        if 'action' in values:
            # Split the text by the Action and Observation markers
            parts = observation.split("Action:\n")
            if len(parts) != 2:
                # No observation found, return None
                return None
            else:
                # Extract the action and observation parts
                action = parts[1].split("Observation:\n")[0].strip()
                observation = parts[1].split("Observation:\n")[1].strip()
                # Return the parsed observation
                return observation
        else:
            return None

# Create an instance of PydanticOutputParser with the MyOutput model as an argument
my_output_parser = PydanticOutputParser(pydantic_object=MyOutput)

# Create an instance of PlanAndExecute agent with my_output_parser as the output_parser argument

agent = PlanAndExecute(planner=planner, executor=executor, output_parser=my_output_parser, verbose=True)

# agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

class MermaidDiagram:
    def __init__(self, mermaid_code,diagram):
        self.mermaid_code = mermaid_code
        self.diagram = diagram

    def generate_mermaid_svg(self):
    # Generate the Mermaid SVG for the diagram
        svg = generate_mermaid_svg(self.mermaid_code, "diagram.svg")
        while svg is None:
        # An error occurred while generating the Mermaid SVG
        # Retry with a different Mermaid code
            cr = st.session_state.cr
            _description = cr.search_code("Explain and analyse the code within 200 words")["answer"]
            use_case_generator = UseCaseGenerator(OPENAI_API_KEY[random.randint(0, len(OPENAI_API_KEY)-1)] , _description, self.diagram)
            use_cases = use_case_generator.generate_use_cases(_description, self.diagram)
            mermaid_diagram = MermaidDiagram(use_cases, self.diagram)
            time.sleep(10)
            svg = mermaid_diagram.generate_mermaid_svg()

            # Provide a download link for the diagram image
        st.markdown(f"Download the diagram: [Link](data:image/svg+xml;base64,{base64.b64encode(svg.encode('utf-8')).decode('utf-8')}), target='diagram.svg', download='diagram.svg'")

            # Display the SVG directly
        pic = st.image(svg, width=600, height=400)
        # use_column_width=True

class DiagramValidator:
    @staticmethod
    def validate_diagram(mermaid_code, diagram_type):
        if diagram_type == "class":
            return DiagramValidator.validate_class_diagram(mermaid_code)
        elif diagram_type == "state":
            return DiagramValidator.validate_state_diagram(mermaid_code)
        elif diagram_type == "sequence":
            return DiagramValidator.validate_sequence_diagram(mermaid_code)
        elif diagram_type == "entity_relationship":
            return DiagramValidator.validate_flowchart_diagram(mermaid_code)
        else:
            return False

    @staticmethod
    def validate_class_diagram(mermaid_code):
        pattern = r'classDiagram\s*\n\s*((?:.+\n)+)'
        match = re.search(pattern, mermaid_code)
        if match:
            classes = match.group(1)
            class_pattern = r'([^\s:\[\]]+)\s*(?::\s*([^[\]]+))?\s*(?:\[\s*(.+)\s*\])?\s*\n'
            return re.findall(class_pattern, classes)
        return None

    @staticmethod
    def validate_state_diagram(mermaid_code):
        pattern = r'stateDiagram\s*\n\s*((?:.+\n)+)'
        match = re.search(pattern, mermaid_code)
        if match:
            transitions = match.group(1)
            transition_pattern = r'([^\s:\[\]]+)\s*(?:-->\s*([^\s:\[\]]+))?\s*(?:\[\s*(.*?)\s*\])?\s*\n'
            return re.findall(transition_pattern, transitions)
        return None

    @staticmethod
    def validate_sequence_diagram(mermaid_code):
        pattern = r'sequenceDiagram\s*\n\s*((?:.+\n)+)'
        match = re.search(pattern, mermaid_code)
        if match:
            interactions = match.group(1)
            interaction_pattern = r'([^\s:\[\]]+)\s*(?:\(([^)]+)\))?\s*(?:-->\s*([^\s:\[\]]+))?\s*(?:\(([^)]+)\))?\s*\n'
            return re.findall(interaction_pattern, interactions)
        return None

    @staticmethod
    def validate_flowchart_diagram(mermaid_code):
        pattern = r'(flowchart|graph)\s*(?:LR|RL|BT|TB)?\s*\n\s*((?:.+\n)+)'
        match = re.search(pattern, mermaid_code)
        if match:
            statements = match.group(2)
            statement_pattern = r'([^\s:\[\]]+)\s*(?:-->\s*([^\s:\[\]]+))?\s*(?:\(([^)]+)\))?\s*\n'
            return re.findall(statement_pattern, statements)
        return None



# Add custom CSS styles
st.markdown(
    """
    <style>
    .chat-container {
        background-color: #f7f7f7;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 3px 5px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    .bot-message {
        background-color: #def1fc;
        color: #00000B;
        padding: 10px 15px;
        border-radius: 5px;
        display: inline-block;
        margin-bottom: 10px;
    }

    .user-message {
        background-color: #e8f4e5;
        padding: 10px 15px;
        border-radius: 5px;
        display: inline-block;
        margin-bottom: 10px;
        text-align: right;
    }
    .pic{
        object-fit:contain;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Get the current session state
session = st.session_state


def main():
    with st.sidebar.form(key="my_form"):
        st.title("Options")
        option = st.radio(
            "Select an option",
            ["Analyse and Explain Code", "View Code's Software Diagrams", "Ask Questions about Vulnerabilities"],
        )

        github_url = st.text_input("Enter GitHub repository URL:")
        git_branch = st.text_input("Enter branch")
        submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            if not github_url:
                st.warning("Please enter a GitHub repository URL.")
                return
            if not git_branch:
                st.warning("Please enter Github branch")
                return
            
            if "cr" not in st.session_state:
                st.session_state.cr = _cr(github_url, git_branch)

            cr = st.session_state.cr
            cr.clear_files()
            cr.generate_documents()
            cr.create_vector_store()

    if option == "Analyse and Explain Code":
        show_code_analyser(github_url)
    elif option == "View Code's Software Diagrams":
        show_code_diagram_flow(github_url)
    elif option == "Ask Questions about Vulnerabilities":
        show_vulnerability_questions(github_url)


def show_code_analyser(github_url):
    st.title("Chat with AI Bot")

    # Initialize conversation history
    if "conversation" not in session:
        session["conversation"] = []

    # with st.expander("Conversation", expanded=False):
    #     display_conversation(session["conversation"])

    # Add a chat container for conversation
    chat_container = st.container()

    with chat_container:
        st.markdown('<div class="bot-message">AI Bot: Welcome! How can I assist you today?</div>', unsafe_allow_html=True)

    # Add a sidebar pane to view code
    with st.expander("View Code", expanded=True):
        if st.button("Fetch Code"):
            fetch_and_display_code(github_url)

    if st.button("Submit Code"):
        output = process_code_analysis()
        session["conversation"].append(("user", github_url))
        session["conversation"].append(("bot", output))

    with chat_container:
        # Display conversation history
        display_conversation(session["conversation"])


def show_code_diagram_flow(github_url):
    st.title("Chat with AI Bot")

    # Initialize conversation history
    if "conversation" not in session:
        session["conversation"] = []

    # Add a chat container for conversation
    chat_container = st.container()

    with chat_container:
        st.markdown('<div class="bot-message">AI Bot: Hello! How can I help you with code diagrams?</div>', unsafe_allow_html=True)

    # Add a sidebar pane to view code
    with st.expander("View Code", expanded=True):
        if st.button("Fetch Code"):
            fetch_and_display_code(github_url)

        # Add diagram selection
        diagram_type = st.selectbox("Select Diagram Type", ["Sequence Diagram", "Class Diagram", "State Chart Diagram"])


    if st.button("Submit Diagram"):
        mermaid_diagram = process_code_diagram(diagram_type)
        # mermaid_html = mermaid_diagram.
        session["conversation"].append(("user", diagram_type))
        # session["conversation"].append(("bot", mermaid_html))

    with chat_container:
        # Display conversation history
        display_conversation(session["conversation"])


def show_vulnerability_questions(github_url):
    st.title("Chat with AI Bot")

    # Initialize conversation history
    if "conversation" not in session:
        session["conversation"] = []

    # Add a chat container for conversation
    chat_container = st.container()

    with chat_container:
        st.markdown('<div class="bot-message">AI Bot: Hi there! What vulnerabilities are you concerned about?</div>', unsafe_allow_html=True)

    # Add a sidebar pane to view code
    with st.expander("View Code", expanded=True):
        if st.button("Fetch Code"):
            fetch_and_display_code(github_url)

    # User query input
    vulnerability_query = st.text_input("Enter your query")

    # Process user input based on selected option
    if st.button("Submit Vulnerability"):
        output = process_vulnerability_questions(vulnerability_query)
        session["conversation"].append(("user", vulnerability_query))
        session["conversation"].append(("bot", output))

    with chat_container:
        # Display conversation history
        display_conversation(session["conversation"])

@st.cache_data(experimental_allow_widgets=True)
def fetch_and_display_code(github_url):
    try:
        # Extract GitHub username and repository name from the URL
        username, repo_name = extract_username_and_repo(github_url)

        # Fetch the code from GitHub API
        url = f"https://api.github.com/repos/{username}/{repo_name}/contents"
        response = requests.get(url)
        if response.status_code == 200:
            files = response.json()
            for file in files:
                if file["type"] == "file":
                    if file["name"].endswith(".pdf"):
                        display_pdf(file["download_url"])
                    elif file["name"].endswith((".mp4", ".avi", ".mov")):
                        display_video(file["download_url"])
                    else:
                        file_content = fetch_file_content(file["download_url"])
                        language = get_file_language(file["name"])
                        st.code(file_content, language=language)
        else:
            st.error("Failed to fetch code from GitHub repository.")

    except ValueError:
        st.error("Invalid GitHub repository URL.")

@st.cache_data
def extract_username_and_repo(github_url):
    # Extract username and repository name from the GitHub URL
    parts = github_url.split("/")
    if len(parts) >= 5 and parts[2] == "github.com":
        username = parts[3]
        repo_name = parts[4]
        return username, repo_name
    else:
        raise ValueError("Invalid GitHub repository URL.")

@st.cache_data
def fetch_file_content(download_url):
    response = requests.get(download_url)
    if response.status_code == 200:
        return response.text
    else:
        raise ValueError("Failed to fetch file content.")


def get_file_language(filename):
    # Extract the file extension and map it to the appropriate programming language
    extension = filename.split(".")[-1]
    language_map = {
        "py": "python",
        "java": "java",
        "cpp": "cpp",
        "c": "c",
        "html": "html",
        "css": "css",
        "js": "javascript",
        "sol": "solidity",
        "rb": "ruby",
        "php": "php",
        "swift": "swift",
        "go": "go",
        "rs": "rust",
        "kt": "kotlin",
        "scala": "scala",
        "lua": "lua",
        "sql": "sql",
    }
    return language_map.get(extension, "plaintext")


def display_pdf(url):
    st.markdown(f'<iframe src="{url}" width="100%" height="5px"></iframe>', unsafe_allow_html=True)


def display_video(url):
    try:
        video = YouTube(url)
        video_stream = video.streams.get_highest_resolution()
        st.video(video_stream.url)
    except Exception as e:
        st.error(f"Failed to display video: {e}")


def process_code_analysis():
    # Process code analysis
    # Implement your logic for code analysis here
    cr = st.session_state.cr
    output = cr.search_code("Explain and Analyse the code within 200 words")["answer"]
    return output


def process_code_diagram(diagram_type):
    # Process code diagram based on the diagram type
    # Implement your logic for code diagram here
    cr = st.session_state.cr
    _description = cr.search_code("Explain and analyse the code within 200 words")["answer"]
    use_case_generator = UseCaseGenerator(OPENAI_API_KEY[random.randint(0, len(OPENAI_API_KEY)-1)] , _description, diagram_type)
    use_cases = use_case_generator.generate_use_cases(_description, diagram_type)
    mermaid_diagram = MermaidDiagram(use_cases, diagram_type)
    mermaid_diagram.generate_mermaid_svg()

def process_vulnerability_questions(query):
    # Process vulnerability questions based on the query
    # Implement your logic for vulnerability questions here
    output = st.session_state.agent(query)
    return output


def display_conversation(conversation):
    for role, message in conversation:
        if role == "bot":
            if isinstance(message, str) and message.startswith("<div"):
                # Message is an HTML string, so render it directly
                st.markdown(message, unsafe_allow_html=True)
            else:
                # Message is not an HTML string, so wrap it in a div
                st.markdown(f'<div class="bot-message">AI Bot: {message}</div>', unsafe_allow_html=True)


# Monkey-patch Streamlit's hashing mechanism to retain state across button clicks
# This ensures that the conversation history and outputs are preserved
if "code_changed" not in session:
    session["code_changed"] = False


if __name__ == "__main__":
    main()

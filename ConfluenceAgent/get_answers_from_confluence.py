from json import load
from atlassian import Confluence
import os

confluence_url = os.getenv("CONFLUENCE_URL")
confluence_key = os.getenv("CONFLUENCE_KEY")
# verify_ssl  = 'PATH_TO_CRT_CERT'
# proxies={"http": "YOUR_ORG_HTTP_PROXY",verify_ssl=verify_ssl, "https": "YOUR_ORG_HTTPS_PROXY"}
confluence = Confluence(url=confluence_url, token = confluence_key)

import yake
from langchain_core.tools import tool
from bs4 import BeautifulSoup

language = "en"
max_ngram_size = 3
deduplication_threshold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 20
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)

@tool
def extract_keywords(user_query: str): 
    """
    This tool is used to extract keywords from a given query.
    """
    return custom_kw_extractor.extract_keywords(user_query)[0]

# @tool
# def search_confluence(keywords: str):
#     """
#     This tool is used to search confluence for the given keywords and get the page ids.
#     """
#     cql_query = f'siteSearch ~ "{keywords}" and type=page'
#     search_response = confluence.cql(cql=cql_query, limit=10)
#     for i in range(0, len(search_response['results'])):
#         id = int( search_response['results'][i]['content']['id'] )
#         body_content = confluence.get_page_by_id(page_id= id, expand='body.storage')['body']['storage']['value']
#         content.append(BeautifulSoup(body_content, "html.parser").get_text() )
#     return content


@tool
def search_confluence(keywords: str):
    """
    This tool is used to search confluence for the given keywords and get the page ids.
    """
    content = []  # Initialize the content list
    try:
        cql_query = f'siteSearch ~ "{keywords}" and type=page'
        search_response = confluence.cql(cql=cql_query, limit=10)
        
        if not search_response or 'results' not in search_response:
            print(f"serach_reposns: {search_response}")
            return "No results found in Confluence"
            
        for result in search_response['results']:
            try:
                page_id = int(result['content']['id'])
                page_content = confluence.get_page_by_id(
                    page_id=page_id, 
                    expand='body.storage'
                )
                body_content = page_content['body']['storage']['value']
                content.append(BeautifulSoup(body_content, "html.parser").get_text())
            except (KeyError, ValueError) as e:
                print(f"Error processing page: {e}")
                continue
                
        return content if content else "No content could be extracted"
        
    except Exception as e:
        return f"Error searching Confluence: {str(e)}"
    
tools = [extract_keywords, search_confluence]

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import GoogleGenerativeAI

llm = GoogleGenerativeAI(model="gemini-2.0-flash")

from langchain.agents import AgentType, initialize_agent

search_agent = initialize_agent(llm= llm, tools=tools, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,handle_parsing_errors=True)

response = search_agent.run("multimodal LLM")

print(response)
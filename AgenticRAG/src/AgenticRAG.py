#Setting up ReAct Agent
from langchain.tools.retriever import create_retriever_tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain_cohere import CohereEmbeddings
from langchain_cohere.llms import Cohere
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
)

persist_directory = "local_vectorstore"
collection_name = "hrpolicy"
PROJECT_ROOT = "<Project Root path>"

vectorstore = Chroma(
    persist_directory=os.path.join(PROJECT_ROOT, "data", persist_directory),
    collection_name=collection_name,
    embedding_function=embeddings,
)

from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore

local_store = "local_docstore"
local_store = LocalFileStore(os.path.join(PROJECT_ROOT, "data", local_store))
docstore = create_kv_docstore(local_store)

child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    parent_splitter=parent_splitter,
    child_splitter=child_splitter
)

from langchain_community.document_loaders import TextLoader

file_path = os.path.join(PROJECT_ROOT, "data/globalcorp_hr_policy.txt")
loader = TextLoader(file_path)
documents = loader.load()


# run only once
vectorstore.persist()
retriever.add_documents(documents, ids=None)

#Given a search query, we need a tool that will return the relevant chunks of the HR document
#The name and description of the tool will be passed in the API call to the LLM, so make sure it is as unambiguous as possible
tool_search = create_retriever_tool(
    retriever=retriever,
    name="search_hr_policy",
    description="Searches and returns excerpts from the HR policy.",
)

#Using an already available  prompt that emphasizes multiple thought-action-observation steps.
from langchain import hub
prompt = hub.pull("hwchase17/react")
print(prompt.template)


llm=Cohere(model="command-nightly",
           temperature=0.75,
           )

#Creating a ReAct agent
from langchain.agents import AgentExecutor, create_react_agent

## Only creates the logical steps 
react_agent = create_react_agent(llm, [tool_search], prompt)

# List of tools to be used
tools = [tool_search]

# Instantiate AgentExecutor that will execute the logical steps that react_agent will generate.
# executes the logical steps we created
agent_executor = AgentExecutor(
agent=react_agent, 
tools=tools,
verbose=True,
handle_parsing_errors=True,
max_iterations = 5 # useful when agent is stuck in a loop
)


query = "Which country has the highest budget?"
print(agent_executor.invoke({"input": query}))

from langchain.agents import  initialize_agent
from langchain.agents import AgentType
from dotenv import load_dotenv
load_dotenv()


from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)

from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
google_search = GoogleSerperAPIWrapper()

tools = [
    Tool(
        name="Intermediate Answer",
        func=google_search.run,
        description="useful for when you need to ask with search"
    )
]

from langchain.agents import initialize_agent
from langchain.agents import AgentType

agent = initialize_agent(tools = tools, 
                         llm = llm, 
                         agent=AgentType.SELF_ASK_WITH_SEARCH, 
                         verbose=True)

# Print the agent's prompt template
print("Agent Prompt Template:")
print(agent.agent.llm_chain.prompt.template)

response = agent.run("What is the hometown of the 2001 US PGA champion?")
print(response)
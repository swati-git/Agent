# Agents

The core idea of agents is to use a language model to choose a sequence of actions to take. In agents, a language model is used as a reasoning engine to determine which actions to take and in which order. 

## Tools and Actions

Agents typically require a set of tools to be specified at the time of their instantiation. 
Be mindful of the choice of tools you provide an agent, as these are the only tools that the agent will use to answer each of the intermediate steps. If it finds a relevant tool — great, it will use it to get the answer. If it doesn't, it will usually iterate a few times (i.e. trying one of the other available tools or its own logical reasoning) and finally return a sub-optimal answer.

## ReAct Implementation

LangChain's implementation of the [ReAct (Reason + Act) agent](https://arxiv.org/abs/2210.03629)
prompts the LLM to generate both reasoning traces and task-specific actions in a step-by-step manner, improving its performance on the task.

## Resources

### Tools
- [Retriever Tool Documentation](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.retriever.create_retriever_tool.html#langchain_core.tools.retriever.create_retriever_tool)

### Dependencies
- [LangChain Package](https://pypi.org/project/langchain/)
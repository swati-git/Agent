# Agent

## Overview
This example is an application that searches for necessary information through the Google search engine and creates an answer by referring to this information in the LLM application. It uses the [Google Search API service from Serper](https://serper.dev/). 

## Features
### Agents with tools:
- Can gather real-time information (weather, stock prices, etc.)
- Can actively interact with external systems (APIs, databases, calculators)

## References
### Model Documentation
- [Claude Models Documentation](https://docs.anthropic.com/en/docs/about-claude/models18)
- [LangChain Anthropic Chat Models](https://python.langchain.com/api_reference/anthropic/chat_models/langchain_anthropic.chat_models.ChatAnthropic.html)

### Package Versions
- [LangChain Community Version](https://github.com/langchain-ai/langchain/blob/master/libs/community/pyproject.toml)
- [LangChain Anthropic Version](https://github.com/langchain-ai/langchain/blob/master/libs/partners/anthropic/pyproject.toml)
- [LangChain Version](https://github.com/langchain-ai/langchain/blob/master/libs/community/pyproject.toml)

## Dependencies
To find compatible versions between various dependencies, check the pyproject.toml files of the packages.

## Agent Types
### SELF_ASK_WITH_SEARCH
A specialized LangChain agent that implements a self-questioning strategy. This agent:

- Implements a recursive self-questioning approach
- Breaks down complex queries into smaller, searchable questions
- Uses the provided search tool (GoogleSerperAPIWrapper) to find answers
- Chains intermediate answers together for the final response

import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent

async def main():
    llm = ChatOpenAI(
        api_key="dummy",
        model="gpt-4o"
    )
    agent = Agent(task="dummy", llm=llm)
    print(dir(agent))

asyncio.run(main())

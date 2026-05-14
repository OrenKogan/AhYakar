import asyncio
import os
from browser_use import Agent, BrowserProfile
from browser_use.llm.openai.chat import ChatOpenAI

async def run_browser_agent():
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        model="openai/gpt-4o"
    )
    
    task_prompt = (
        "You are an autonomous medical booking agent. Your strict task is to book a 'Family Doctor' appointment.\n"
        "1. Go to http://127.0.0.1:7800/.\n"
        "2. You MUST log in. Type 'a@gmail.com' into the email input, type '123123' into the password input, and click the Login button.\n"
        "3. Wait for the dashboard to load, then click on the 'Find Doctors' navigation tab.\n"
        "4. Find the specialty filters and click the pill that best matches 'Family Doctor'.\n"
        "5. Click the 'Book' button for the first doctor in the results list.\n"
        "6. In the booking modal, select tomorrow's date. If unable to find a time slot for tomorrow, pick the next available day and select a time slot.\n"
        "7. Click the 'Confirm Booking' button.\n"
        "8. DO NOT complete the task until you actually see the confirmation screen. Once confirmed, extract the doctor's name, the clinic location, and the booked time slot, and return them."
    )
    
    profile = BrowserProfile(
        args=['--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage', '--disable-software-rasterizer']
    )

    agent = Agent(task=task_prompt, llm=llm, browser_profile=profile)
    result = await agent.run()
    print("FINAL RESULT:", result.final_result() if result else "None")

asyncio.run(run_browser_agent())

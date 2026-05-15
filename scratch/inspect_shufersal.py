import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print("Navigating to Shufersal...")
        await page.goto("https://www.shufersal.co.il/online/he/search?text=אקמול")
        await page.wait_for_selector('.product-item, .miglog-prod-full', timeout=10000)
        
        # Get the first product card HTML
        card = await page.query_selector('.product-item, .miglog-prod-full')
        if card:
            html = await card.inner_html()
            print("--- PRODUCT CARD HTML ---")
            print(html)
            print("--- END ---")
        else:
            print("No card found")
        
        await browser.close()

asyncio.run(run())

"""
Pharmacy Scraper — Vision-Based Approach
=========================================
Instead of fragile CSS selectors that break every time sites update,
this scraper:
1. Opens the pharmacy search page with Playwright
2. Takes a full-page screenshot
3. Sends it to a Vision LLM (via OpenRouter) to extract products & prices
4. Finds the cheapest valid result
"""

import asyncio
import base64
import json
import logging
import os
import re
import sys
import uuid
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from playwright.async_api import async_playwright, Page

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PharmacyScraper")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

PHARMACY_SITES = [
    {
        "name": "Super-Pharm",
        "url_template": "https://shop.super-pharm.co.il/search?av={query}",
        "base_url": "https://shop.super-pharm.co.il",
        "wait_seconds": 5,
    },
    {
        "name": "E-Pharma",
        "url_template": "https://www.epharma.co.il/search?q={query}",
        "base_url": "https://www.epharma.co.il",
        "wait_seconds": 7,
    },
]



async def take_screenshot(url: str, wait_seconds: int = 4) -> bytes | None:
    """Open a page with Playwright, scroll to load content, and capture a screenshot."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
            )
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1400, "height": 900},
                locale="he-IL",
            )
            page: Page = await context.new_page()

            logger.info(f"Navigating to {url}")
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=25000)
            except Exception:
                logger.warning(f"Timeout navigating to {url}, taking screenshot anyway")

            # Wait for initial load
            await asyncio.sleep(wait_seconds)
            
            # Slow scroll to trigger all lazy loading
            logger.info(f"Scrolling to capture all items on {url}")
            await page.evaluate("""
                async () => {
                    await new Promise((resolve) => {
                        let totalHeight = 0;
                        let distance = 300;
                        let timer = setInterval(() => {
                            let scrollHeight = document.body.scrollHeight;
                            window.scrollBy(0, distance);
                            totalHeight += distance;
                            if(totalHeight >= scrollHeight){
                                clearInterval(timer);
                                resolve();
                            }
                        }, 150);
                    });
                }
            """)
            await asyncio.sleep(1) # Final settle
            
            # Try to close common pop-ups
            try:
                await page.keyboard.press("Escape")
                close_selectors = ["button[aria-label='Close']", ".close", ".modal-close", "a[title='Close']", ".pop-up-close"]
                for selector in close_selectors:
                    if await page.is_visible(selector):
                        await page.click(selector)
            except Exception:
                pass
            
            await page.evaluate("window.scrollTo(0, 0)") # Back to top for cleaner capture? No, full_page=True handles it.
            await asyncio.sleep(0.5)

            screenshot = await page.screenshot(full_page=True)
            
            # Save screenshot for debugging
            os.makedirs("debug_screenshots", exist_ok=True)
            safe_url = url.replace("https://", "").replace("/", "_").replace("?", "_").replace("=", "_")[:100]
            with open(f"debug_screenshots/{safe_url}.png", "wb") as f:
                f.write(screenshot)
            
            await browser.close()
            logger.info(f"Full-page screenshot taken for {url} ({len(screenshot)} bytes)")
            return screenshot
    except Exception as e:
        logger.error(f"Failed to screenshot {url}: {e}")
        return None


async def analyze_screenshot_with_vision(
    screenshot_bytes: bytes,
    search_term: str,
    pharmacy_name: str,
    pharmacy_url: str,
) -> list[dict]:
    """
    Send screenshot to Vision LLM and extract product list with prices.
    Returns a list of dicts: [{"product_name": str, "price": float, "pharmacy": str, "url": str}]
    """
    if not OPENROUTER_API_KEY:
        logger.error("No OPENROUTER_API_KEY set")
        return []

    image_b64 = base64.standard_b64encode(screenshot_bytes).decode("utf-8")

    prompt = f"""You are analyzing a screenshot of an Israeli pharmacy website called "{pharmacy_name}".
The user searched for: "{search_term}"

Look at the visible product cards on the page and extract ALL products clearly related to "{search_term}".
For each product, extract:
1. The full product name (in Hebrew if shown, exactly as displayed)
2. The quantity/size (e.g. "20 caps", "100 ml", "500 mg") if visible
3. The price in NIS (₪) — just the number, e.g. 14.24
4. The direct product URL if visible in the page. If not visible, use this search page URL: {pharmacy_url}

IMPORTANT rules:
- Only include products clearly related to "{search_term}"
- Ignore products with no visible price
- Return ONLY valid JSON, no markdown, no explanation

Return this exact JSON format:
[
  {{"product_name": "...", "quantity": "...", "price": 14.24, "pharmacy": "{pharmacy_name}", "url": "..."}},
  ...
]

If no relevant products are visible, return: []
"""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "google/gemini-2.0-flash-001",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_b64}"
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    "max_tokens": 2000,
                },
            )

        if response.status_code != 200:
            logger.error(f"Vision API error {response.status_code}: {response.text}")
            return []

        content = response.json()["choices"][0]["message"]["content"].strip()
        logger.info(f"Vision response from {pharmacy_name}: {content[:200]}")

        # Robustly extract JSON array from response (handles markdown fences, extra text)
        content = content.replace("```json", "").replace("```", "").strip()
        
        # Find the JSON array bounds
        start = content.find('[')
        end = content.rfind(']') + 1
        
        if start == -1:
            logger.warning(f"No JSON array start found in vision response from {pharmacy_name}. Content: {content[:500]}")
            return []
            
        if end == 0:
            # Maybe it was truncated? Try to fix it if it looks like it's in the middle of a list
            logger.warning(f"No JSON array end found in vision response from {pharmacy_name}. Attempting to fix truncated JSON.")
            content = content[start:]
            if not content.endswith(']'):
                # Very basic fix: append ] if it seems to be missing
                content += ']'
            # Re-find end
            end = content.rfind(']') + 1
            if end == 0:
                return []
        
        content = content[start:end]

        products = json.loads(content)
        if isinstance(products, list):
            # Validate and normalize
            valid = []
            for p in products:
                if isinstance(p, dict) and "product_name" in p and "price" in p:
                    try:
                        p["price"] = float(p["price"])
                        p["pharmacy"] = pharmacy_name
                        if "url" not in p or not p["url"]:
                            p["url"] = pharmacy_url
                        p["quantity"] = p.get("quantity", "N/A")
                        p["is_in_stock"] = True
                        valid.append(p)
                    except (ValueError, TypeError):
                        pass
            logger.info(f"Extracted {len(valid)} products from {pharmacy_name}")
            return valid
        return []

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error from vision for {pharmacy_name}: {e}")
        return []
    except Exception as e:
        logger.error(f"Vision API call failed for {pharmacy_name}: {e}")
        return []


async def scrape_pharmacy_with_vision(pharmacy: dict, search_term: str) -> tuple[list[dict], str | None]:
    """Scrape a single pharmacy using screenshot + vision. Returns (products, screenshot_path)"""
    encoded = quote(search_term)
    url = pharmacy["url_template"].format(query=encoded)
    wait = pharmacy.get("wait_seconds", 4)

    screenshot = await take_screenshot(url, wait_seconds=wait)
    if not screenshot:
        logger.warning(f"No screenshot for {pharmacy['name']}, skipping.")
        return [], None

    # Save screenshot to a permanent location for the UI
    os.makedirs("uploads/search_results", exist_ok=True)
    filename = f"{pharmacy['name'].lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join("uploads/search_results", filename)
    with open(filepath, "wb") as f:
        f.write(screenshot)

    products = await analyze_screenshot_with_vision(
        screenshot_bytes=screenshot,
        search_term=search_term,
        pharmacy_name=pharmacy["name"],
        pharmacy_url=url,
    )
    
    # Attach screenshot path to products
    for p in products:
        p["screenshot_path"] = f"/uploads/search_results/{filename}"
        
    return products, f"/uploads/search_results/{filename}"


async def scrape_all_pharmacies(search_term: str) -> list[dict]:
    """Scrape all pharmacies in parallel using vision and return sorted results."""
    logger.info(f"Starting vision-based scrape for: {search_term}")

    tasks = [scrape_pharmacy_with_vision(p, search_term) for p in PHARMACY_SITES]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_products: list[dict] = []
    pharmacy_screenshots: dict[str, str] = {}
    
    for i, result in enumerate(results):
        pharmacy_name = PHARMACY_SITES[i]["name"]
        if isinstance(result, tuple):
            products, screenshot_path = result
            all_products.extend(products)
            if screenshot_path:
                pharmacy_screenshots[pharmacy_name] = screenshot_path
        elif isinstance(result, Exception):
            logger.error(f"Scrape task failed for {pharmacy_name}: {result}")

    # Sort by price ascending (cheapest first)
    all_products.sort(key=lambda x: x.get("price", 9999))

    # Save to JSON
    with open("otc_medications_db.json", "w", encoding="utf-8") as f:
        json.dump(all_products, f, ensure_ascii=False, indent=4)

    logger.info(f"Done. Saved {len(all_products)} products to otc_medications_db.json")
    if all_products:
        best = all_products[0]
        logger.info(f"Cheapest: {best['product_name']} at {best['pharmacy']} for ₪{best['price']}")

    return all_products, pharmacy_screenshots


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    term = sys.argv[1] if len(sys.argv) > 1 else "דקסמול"
    asyncio.run(scrape_all_pharmacies(term))

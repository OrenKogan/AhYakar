from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("http://127.0.0.1:7800/")
    
    try:
        # Check if login screen is visible
        if page.locator('#login-screen').is_visible():
            page.fill('#login-id', '123456789')
            page.click('#btn-login')
            page.wait_for_timeout(1000)
    except:
        pass
        
    page.click('div[data-page="doctors"]')
    page.wait_for_timeout(500)
    page.fill('#doctor-search', 'Cardiologist')
    page.wait_for_timeout(500)
    
    print("Doctor cards:")
    for card in page.locator('.doctor-card').all():
        print(card.inner_html())
        break
        
    browser.close()

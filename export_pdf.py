import asyncio
import os
from playwright.async_api import async_playwright

async def export_poster():
    # Make sure to run this inside the same folder as poster.html
    html_path = f"file:///{os.path.abspath('poster.html').replace(os.sep, '/')}"
    
    print(f"Loading {html_path}...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Go to the local poster HTML file
        await page.goto(html_path, wait_until="networkidle")
        
        # PDF Generation
        # Dimensions set exactly to the pull-up banner constraints
        print("Generating PDF at 800mm x 2000mm...")
        await page.pdf(
            path="poster.pdf",
            width="800mm",
            height="2000mm",
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"}
        )
        
        await browser.close()
        print("Success! poster.pdf created.")

if __name__ == "__main__":
    asyncio.run(export_poster())

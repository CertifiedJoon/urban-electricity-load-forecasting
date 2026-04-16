import asyncio
from playwright.async_api import async_playwright
from PIL import Image

async def export_poster_png():
    print("Loading http://localhost:5173/ ...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        
        # Physical rolling banner size: 800mm x 2090mm
        # We set the base viewport exactly to the CSS container bounds: 800px by 2090px.
        # We use a device_scale_factor of 4.92 (increased by 23%) to natively upsample the output to a massive 3936x10282 PNG.
        context = await browser.new_context(
            viewport={"width": 800, "height": 2090},
            device_scale_factor=4.92
        )
        page = await context.new_page()
        
        await page.goto("http://localhost:5173/", wait_until="networkidle")
        
        # 1. Inject CSS to scale up `rem` units for massive 2985px display
        await page.add_style_tag(content="""
            html { font-size: 38px !important; }
            .hero-section {
                min-height: auto !important;
                padding-top: 5rem !important;
                padding-bottom: 3rem !important;
            }
        """)

        # 2. Scroll to the bottom to trigger all framer-motion 'whileInView' events
        await page.evaluate("""
            window.scrollTo(0, document.body.scrollHeight);
        """)
        await asyncio.sleep(1) # wait for animations to play
        
        # 3. Clean up any remaining framer-motion inline styles just in case
        await page.evaluate("""
            document.querySelectorAll('div').forEach(el => {
                if (el.style.opacity) el.style.opacity = '1';
                if (el.style.transform && el.style.transform.includes('translate')) {
                    el.style.transform = 'none';
                }
            });
            window.scrollTo(0, 0);
        """)
        
        await asyncio.sleep(2)
        
        print("Generating PNG snapshot scaled natively for 800mm x 2090mm printing...")
        
        # Look for the exact outer padding wrapper to cleanly screenshot
        wrapper = await page.query_selector("div[style*='800px']")
        if wrapper:
            await wrapper.screenshot(path="poster-react.png", type="png")
        else:
            await page.screenshot(path="poster-react.png", full_page=True, type="png")
        
        await browser.close()
        
        print("Enforcing explicit 150 DPI configuration metadata...")
        img = Image.open("poster-react.png")
        img.save("poster-react.png", dpi=(150, 150))
        
        print("Success! High-resolution poster-react.png created directly from the Chromium engine.")

if __name__ == "__main__":
    asyncio.run(export_poster_png())

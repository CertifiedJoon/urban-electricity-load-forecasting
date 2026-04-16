import asyncio
from playwright.async_api import async_playwright

async def export_poster():
    print("Loading http://localhost:5173/ ...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # We set viewport to the target PDF layout size (800mm x 2090mm at 96 DPI)
        target_width = 3023
        target_height = 7899
        await page.set_viewport_size({"width": target_width, "height": target_height})
        
        await page.goto("http://localhost:5173/", wait_until="networkidle")
        
        # 1. Inject CSS to scale up `rem` units for massive 2985px display
        # and override the hero section's 100vh so it doesn't take up the whole 4913px height.
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
        
        await asyncio.sleep(1)
        
        # 4. Body scale logic using zoom to avoid transform conflicts
        await page.evaluate(f"""
            const contentHeight = document.body.scrollHeight;
            const targetHeight = {target_height};
            if (contentHeight > targetHeight) {{
                const scale = targetHeight / contentHeight;
                document.body.style.zoom = scale;
            }}
        """)

        await asyncio.sleep(2)
        
        print("Generating PDF at 800mm x 2090mm...")
        await page.pdf(
            path="poster-react.pdf",
            width="800mm",
            height="2090mm",
            print_background=True,
            page_ranges="1", 
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"}
        )
        
        await browser.close()
        print("Success! poster-react.pdf created.")

if __name__ == "__main__":
    asyncio.run(export_poster())

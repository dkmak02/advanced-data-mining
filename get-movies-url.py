import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import os
import aiofiles

base_url = "https://letterboxd.com/films/popular/page/{}/"
os.makedirs("data", exist_ok=True)
csv_path = "data/movies.csv"

if not os.path.exists(csv_path):
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write('"Title","URL"\n')

async def fetch_page(browser, page_num):
    page = await browser.new_page()
    url = base_url.format(page_num)
    print(f"🌐 Getting page {page_num}")
    await page.goto(url)
    await page.wait_for_selector("ul.poster-list")
    html = await page.content()
    await page.close()

    soup = BeautifulSoup(html, "html.parser")
    movie_list = soup.find("ul", class_="poster-list")
    if not movie_list:
        return []

    movies = []
    for item in movie_list.find_all("li"):
        link_tag = item.find("a", href=True)
        if link_tag:
            title = link_tag.get("title") or link_tag.text.strip()
            href = link_tag["href"]
            full_url = f"https://letterboxd.com{href}"
            movies.append((title, full_url))
    return movies

async def save_movies(movies):
    async with aiofiles.open(csv_path, "a", encoding="utf-8") as f:
        for title, url in movies:
            await f.write(f'"{title}","{url}"\n')

async def worker(semaphore, browser, page_num):
    async with semaphore:
        movies = await fetch_page(browser, page_num)
        if movies:
            await save_movies(movies)

async def main():
    semaphore = asyncio.Semaphore(10)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        tasks = [worker(semaphore, browser, i) for i in range(1, 151)]
        await asyncio.gather(*tasks)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())

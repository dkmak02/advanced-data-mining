import asyncio
import json

import aiofiles
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import os

input_csv_path = "data/movies.csv"
output_json_path = "data/movies_data.json"
os.makedirs("data", exist_ok=True)

# --- Initialize JSON array ---
with open(output_json_path, "w", encoding="utf-8") as f:
    f.write("[\n")

async def fetch_movie_data(browser, title, url):
    page = await browser.new_page()
    print(f"🎬 Scraping: {title} -> {url}")
    await page.goto(url)
    await page.wait_for_selector("span.name", timeout=10000)
    html = await page.content()
    await page.close()

    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1", class_="primaryname")
    title_page = title_tag.text.strip() if title_tag else title

    year_tag = soup.find("span", class_="releasedate")
    year = year_tag.text.strip() if year_tag else ""

    genre_tags = soup.select('div#tab-genres a')
    genres = ", ".join([g.text for g in genre_tags]) if genre_tags else ""

    description_tag = soup.find("div", class_="truncate")
    description = description_tag.text.strip() if description_tag else ""

    review_divs = soup.select(".js-popular-reviews div.js-review-body")
    all_reviews = [div.get_text(separator="\n").strip() for div in review_divs if div.get_text(strip=True)]

    return {
        "Title": title_page,
        "URL": url,
        "Year": year,
        "Genres": genres,
        "Description": description,
        "AllReviews": all_reviews
    }

# --- Save each movie with comma ---
async def save_movie(movie, is_last=False):
    async with aiofiles.open(output_json_path, "a", encoding="utf-8") as f:
        json_text = json.dumps(movie, ensure_ascii=False, indent=2)
        if not is_last:
            json_text += ",\n"
        else:
            json_text += "\n"
        await f.write(json_text)

# --- Worker ---
async def worker(semaphore, browser, title, url, results, is_last=False):
    async with semaphore:
        try:
            movie = await fetch_movie_data(browser, title, url)
            await save_movie(movie, is_last=is_last)
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")

# --- Main ---
async def main():
    semaphore = asyncio.Semaphore(5)
    movies = []

    import csv
    with open(input_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                movies.append((row[0], row[1]))

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        tasks = [
            worker(semaphore, browser, title, url, movies, is_last=(i == len(movies)-1))
            for i, (title, url) in enumerate(movies)
        ]
        await asyncio.gather(*tasks)
        await browser.close()

    # Close JSON array
    async with aiofiles.open(output_json_path, "a", encoding="utf-8") as f:
        await f.write("]")

if __name__ == "__main__":
    asyncio.run(main())

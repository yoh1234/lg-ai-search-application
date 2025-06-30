from lg_scraper import LGProductScraper

categories = {
    "TVs": "https://www.lg.com/us/tvs",
    # "Monitors": "https://www.lg.com/us/monitors",
    # "Soundbars": "https://www.lg.com/us/soundbars",
    # "TVs_Accessories": "https://www.lg.com/us/tv-audio-video-accessories"
    # "Monitors_Accessories": "https://www.lg.com/us/computer-accessories"
}

for cat_name, url in categories.items():
    scraper = LGProductScraper(category_url=url, category_name=cat_name)
    scraper.scrape()
    scraper.close()


"""
    get_category_items.py

    MediaWiki API Demos
    Demo of `Categorymembers` module : List twenty items in a category

    MIT License
"""
import os
import requests
from pprint import pprint

# os.rmdir("data")
os.mkdir("data")
CATEGORIES = ["Physics", "Culture", "Geography"]
# CATEGORIES = ["Physics"]

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

for category in CATEGORIES:
    params = {
        "action": "query",
        "cmtitle": "Category:" + category,
        "cmlimit": "20",
        "list": "categorymembers",
        "cmtype": "page",
        "format": "json"
    }

    R = S.get(url=URL, params=params)
    DATA = R.json()

    os.mkdir("data/" + category)

    PAGES = DATA['query']['categorymembers']

    for page in PAGES:
        page_params = {
            "action": "parse",
            "pageid": page["pageid"],
            "prop": "wikitext",
            "format": "json"
        }

        page_request = S.get(url=URL, params=page_params)
        page_data = page_request.json()

        file = open("data/" + category + "/"+ str(page["pageid"]) + ".txt", "a")
        for line in page_data['parse']['wikitext']:
            file.write(page_data['parse']['wikitext'][line])
        file.close()

        # print(page['title'])
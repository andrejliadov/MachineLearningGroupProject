
"""
    get_category_items.py

    MediaWiki API Demos
    Demo of `Categorymembers` module : List twenty items in a category

    MIT License
"""
import os
import requests
from pprint import pprint
import math

# os.rmdir("data")
os.mkdir("data")
CATEGORIES = ["Physics", "Culture", "Geography"]
# CATEGORIES = ["Physics"]

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

data_points_per_category = 500;

def get_category_data(category, num, top_level_category):
    params = {
        "action": "query",
        "cmtitle": "Category:" + category,
        "cmlimit": str(num),
        "list": "categorymembers",
        "cmtype": "page",
        "format": "json"
    }

    R = S.get(url=URL, params=params)
    data = R.json()

    if not os.path.isdir("data/" + top_level_category):
        os.mkdir("data/" + category)

    pages = data['query']['categorymembers']
    pprint(pages)

    for page in pages:
        if not os.path.isfile("data/" + top_level_category + "/" + str(page["pageid"])):

            page_params = {
                "action": "parse",
                "pageid": page["pageid"],
                "prop": "wikitext",
                "format": "json"
            }

            page_request = S.get(url=URL, params=page_params)
            page_data = page_request.json()

            file = open("data/" + top_level_category + "/"+ str(page["pageid"]) + ".txt", "a")
            for line in page_data['parse']['wikitext']:
                file.write(page_data['parse']['wikitext'][line])
            file.close()

    if len(pages) < num:
        params["cmtype"] = "subcat"
        params["num"] = num - len(pages)

        R = S.get(url=URL, params=params)
        data = R.json()
        categories = data['query']['categorymembers']
        print("num categories in " + category + ": " + str(len(categories)))
        for subcategory in categories:
            print((num - len(pages)) / len(categories))
            subcat_title = subcategory["title"].replace("Category:", "")
            pprint(subcat_title)
            get_category_data(subcat_title, math.floor((num - len(pages)) / len(categories)), top_level_category)




for category in CATEGORIES:
    get_category_data(category, data_points_per_category, category)
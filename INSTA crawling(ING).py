#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from lib2to3.pgen2 import driver
from bs4 import BeautifulSoup
from selenium import webdriver
import urllib.request
from time import sleep


# url 경로로 다운로드

def downImage(url, name):
    dir = 'C:\\Users\\USER\\Sad\\'

    urllib.request.urlretrieve(url, dir + name + '.jpg')


#검색어를 입력하세요

hashtag = ""
baseurl = f"https://www.instagram.com/explore/tags/{hashtag}/"
plusurl = ""

browser = webdriver.Chrome(r"C:\Users\USER\Desktop\ksa_ai\chromedriver.exe")

browser.get(url)

pic_name = browser.find_elements_by_class_name("rg_i")

flg = 0

InstaImg
InstaText
InstaHashtag


# 자동 스크롤링

last_height = browser.execute_script("return document.body.scrollHeight")
while True:

    # Scroll down to bottom
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    # Wait to load page
    sleep(2)
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight-50);")
    sleep(2)
    # Calculate new scroll height and compare with last scroll height
    new_height = browser.execute_script("return document.body.scrollHeight")

    if new_height == last_height:
        break

    last_height = new_height
#
for idx, value in enumerate(browser.find_elements_by_class_name("rg_i.Q4LuWd")):

    print(idx, "번째")

    # value.screenshot(search_term + "_" + str(idx) + ".png")

    if value.get_attribute("data-src") == None:

        img_url[idx] = value.get_attribute("src")



    else:

        img_url[idx] = value.get_attribute("data-src")

    # print(value.get_attribute("data-src"))

    # screen_data[idx] = value.get_attribute("data-src")

    print(img_url[idx])

    downImage(img_url[idx], 'sad08' + str(idx))

    if flg == 1000:
        break

    flg += 1


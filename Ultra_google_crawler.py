#!/usr/bin/env python
# coding: utf-8

# In[12]:


from lib2to3.pgen2 import driver

from selenium import webdriver

import urllib.request

from time import sleep


# url 경로로 다운로드

def downImage(url, name):
    dir = 'C:\\Users\\USER\\Sad\\'

    urllib.request.urlretrieve(url, dir + name + '.jpg')


#search_term = '우는 사진'
#url= f"https://www.google.com/search?q={search_term}&tbm=isch&ved="

url = "https://www.google.com/search?q=crying+scene&tbm=isch&ved=2ahUKEwjz56aA9f3qAhUMfZQKHeM9AEQQ2-cCegQIABAA&oq=crying+scene&gs_lcp=CgNpbWcQAzIECAAQEzIECAAQEzIECAAQEzIECAAQEzIECAAQEzIECAAQEzIECAAQEzIECAAQEzIECAAQEzIGCAAQHhATOgIIADoFCAAQsQM6BAgAEB5QuUhY_mpgu2xoAnAAeACAAawBiAGSCpIBBDEzLjGYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=i2wnX7P2Noz60QTj-4CgBA&bih=792&biw=1519&rlz=1C1CAFC_enKR901KR901&hl=ko&hl=ko"
browser = webdriver.Chrome(r"C:\Users\USER\Desktop\ksa_ai\chromedriver.exe")

browser.get(url)

pic_name = browser.find_elements_by_class_name("rg_i")
flg = 0
img_url = {}
# url 경로 모음.

#scroller_with_selenium

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

    
    #image 갯수
    if flg == 1000:
        break

    flg += 1


# In[ ]:





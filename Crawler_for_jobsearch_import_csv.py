import os
import csv
import requests
from bs4 import BeautifulSoup

os.system("clear")
alba_url = "http://www.alba.co.kr"

def company_list(url):
    result=requests.get(url)
    soup=BeautifulSoup(result.text,"html.parser")
    results=soup.find_all("li",{"class":"impact"})
    companies_link= []
   
    for result in results:
        com_lnk = result.find("a")["href"]
        companies_link.append(com_lnk)

    for company in companies_link:
        job_list(company)

def job_list(link):
     alba = []
     result=requests.get(f"{link}")
     soup=BeautifulSoup(result.text,"html.parser")
     results = soup.find("tbody").find_all("tr")
#    results=soup.find_all("tbody",{"tr class":None})
#    print(results)
     company =link.split(".")[0].split("//")[1]
     for result in results[0::2]:
   #    print(result)
        if {"class":"summary view"} in result:
            pass
        else:
            place = result.find("td", {"class": "local first"}).text
            title=result.find("span",{"class": "title"}).text
            time=result.find("span", {"class": "time"}).text
            pay=result.find("td", {"class": "pay"}).text
            date=result.find("td", {"class": "regDate last"}).text
            list={"company":company,"place":place,"title":title,"time":time,"pay":pay,"date":date}
            alba.append(list)
     save(alba)

def save(jobs):
    title= jobs[0]["company"]
    file = open(f"{title}.csv", "w", -1, "utf-8")
    #print(len(jobs))
    writer = csv.writer(file)
    writer.writerow(["place", "title", "time", "pay", "date"])
    for i in range(len(jobs)):
        writer.writerow(list(jobs[i].values()))
    return



company_list(alba_url)

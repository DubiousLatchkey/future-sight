import requests
import json 
import os
from requests_html import HTMLSession

def getReplays(pageNumber):
    battleLinks = []
    session = HTMLSession()
    response = session.get("https://replay.pokemonshowdown.com/?format=%5Bgen9-randombattle%5D&page=" + str(pageNumber))
    response.html.render(timeout=20, sleep=2)
    #print(response.html.links)
    for link in response.html.links:
        if("gen9randombattle-" in link):
            battleLinks.append(link)
    
    return battleLinks

def getAllReplays():
    pageNumber = 1
    allLinks = []
    while(True):
        links = getReplays(pageNumber=pageNumber)
        if(len(links) == 0):
            return allLinks
        allLinks += links
        pageNumber+= 1
        print("Replays found:", len(allLinks))

replayIds = getAllReplays()
if(not os.path.exists("onlineReplays/")):
        os.mkdir("onlineReplays/")

for replayId in replayIds:
    if(not os.path.exists("onlineReplays/" + replayId + ".json")):
        replay = requests.get("https://replay.pokemonshowdown.com/" + replayId +".json")
        try:
            data = replay.json()
            with open("onlineReplays/" + replayId + ".json", "w+") as f:
                json.dump(data, f, indent=4)
            print("Saved replay:", replayId)
        except:
            print("error saving replay", replayId)
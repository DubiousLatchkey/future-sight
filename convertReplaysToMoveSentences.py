import json
import os
import re

def processMoveNameIntoID(move):
    return move.lower().replace(" ", "").replace("-", "").replace(",", "")

if(not os.path.exists("onlineReplayMoveSentences/") ):
    os.mkdir("onlineReplayMoveSentences")

count = 0
for filename in os.listdir("onlineReplays/"):
    if( filename[:5] == "gen9r"):
        with open("onlineReplays/" + filename) as f:
            data = json.load(f)
            log = str(data["log"])

            moves = [processMoveNameIntoID(line.split("|")[3]) for line in log.split("\n") if line[:6] == "|move|"]

            if(not os.path.exists("onlineReplayMoveSentences/" + filename + ".txt") ):
                with open("onlineReplayMoveSentences/" + filename + ".txt", "w") as outputFile:
                    outputFile.write(" ".join(moves))
                    count += 1

                    if(count % 100 == 0):
                        print("Processed", count)
            
# with open("venv\Lib\site-packages\poke_env\data\static\moves\gen9moves.json") as f:
#     data = json.load(f)

#     for key in data.keys():
#         if(re.search("([^a-z A-Z])", key)):
#             print(key)
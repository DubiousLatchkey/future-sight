import json
import os
import re

def processPokemonName(pokemon):
    return pokemon.lower().replace(" ", "").replace("-", "").replace(",", "")

if(not os.path.exists("onlineReplayTeams/") ):
    os.mkdir("onlineReplayTeams")

count = 0

for foldername in os.listdir("gameStates/"):
    with open(os.path.join("gameStates/", foldername, "MinimaxPlayer 1_1.json") ) as f:
        data = json.load(f)
        team = data["team"]
        player1Team = [pokemon["species"] for pokemon in team]

        if(not os.path.exists("onlineReplayTeams/" + foldername + "_1.txt") ):
            with open("onlineReplayTeams/" + foldername + "_1.txt", "w") as outputFile:
                outputFile.write(" ".join(player1Team ))
    
    with open(os.path.join("gameStates/",foldername, "MinimaxPlayer 2_1.json") ) as f:
        data = json.load(f)
        team = data["team"]
        player2Team = [pokemon["species"] for pokemon in team]

        if(not os.path.exists("onlineReplayTeams/" + foldername + "_2.txt") ):
            with open("onlineReplayTeams/" + foldername + "_2.txt", "w") as outputFile:
                outputFile.write(" ".join(player2Team ))
                
                count += 1

                if(count % 100 == 0):
                    print("Processed", count)


# for filename in os.listdir("onlineReplays/"):
#     if( filename[:5] == "gen9r"):
#         player1Team = {}
#         player2Team = {}
#         with open("onlineReplays/" + filename) as f:
#             data = json.load(f)
#             log = str(data["log"])

#             player1Results = re.findall("(?<=switch\|p1a: )[^|]+(?=\|)", log)
#             player2Results = re.findall("(?<=switch\|p2a: )[^|]+(?=\|)", log)

#             player1Results = set(player1Results)
#             player2Results = set(player2Results)

#             if(not os.path.exists("onlineReplayTeams/" + filename[:-5] + "_1.txt") ):
#                 with open("onlineReplayTeams/" + filename[:-5] + "_1.txt", "w") as outputFile:
#                     outputFile.write(" ".join(map(processPokemonName, list(player1Results)) ))

#             if(not os.path.exists("onlineReplayTeams/" + filename[:-5] + "_2.txt") ):
#                 with open("onlineReplayTeams/" + filename[:-5] + "_2.txt", "w") as outputFile:
#                     outputFile.write(" ".join(map(processPokemonName, list(player2Results)) ))
            
#                 count += 1

#                 if(count % 100 == 0):
#                     print("Processed", count)


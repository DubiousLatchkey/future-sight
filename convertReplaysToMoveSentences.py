import json
import os
import re

def processMoveNameIntoID(move):
    return move.lower().replace(" ", "").replace("-", "").replace(",", "")

if(not os.path.exists("onlineReplayMoveSentences/") ):
    os.mkdir("onlineReplayMoveSentences")

# Create pokemon dictinoary
pokemonDict = {}
with open("venv\Lib\site-packages\poke_env\data\static\pokedex\gen9pokedex.json") as f:
    data = json.load(f)

    for id in data.keys():
        pokemon_name = data[id]["name"]
        pokemonDict[pokemon_name] = id
        


def extract_move_sequence(replay):
    """
    Extracts an ordered list of moves from the replay's log.
    Each move is formatted as "PokemonName-MoveName".
    
    Parameters:
        replay (dict): The replay JSON loaded as a dictionary.
    
    Returns:
        list: A list of strings representing the move sequence.
    """
    action_sequence = []
    
    # Get the log text (if not present, use an empty string)
    log_text = replay.get("log", "")
    
    # Process the log line by line
    for line in log_text.splitlines():
        # We only care about lines that start with "|move|"
        if line.startswith("|move|"):
            # Split the line by the pipe character.
            # Example line:
            #   |move|p1a: Slowbro|Psyshock|p2a: Gouging Fire
            # After splitting, tokens[1] == "move", tokens[2] is the attacker info,
            # tokens[3] is the move name.
            tokens = line.split("|")
            if len(tokens) >= 4:
                attacker_token = ''.join(tokens[2].strip().split(" ", 1)[1:])  # e.g., "p1a: Slowbro" -> "Slowbro"
                move_name = processMoveNameIntoID(tokens[3])         # e.g., "Psyshock" -> psyshock
                
                # Combine into the desired format and append to our sequence.
                action_sequence.append(pokemonDict[attacker_token])
                action_sequence.append(move_name)
                
    return action_sequence


count = 0
for filename in os.listdir("onlineReplays/"):
    if( filename[:5] == "gen9r"):
        with open("onlineReplays/" + filename) as f:
            data = json.load(f)
            #log = str(data["log"])

            # moves = [processMoveNameIntoID(line.split("|")[3]) for line in log.split("\n") if line[:6] == "|move|"]
            moves = extract_move_sequence(data)

            if(not os.path.exists("onlineReplayMoveSentences/" + filename + ".txt") ):
                with open("onlineReplayMoveSentences/" + filename[:-5] + ".txt", "w") as outputFile:
                    outputFile.write(" ".join(moves))
                    count += 1

                    if(count % 100 == 0):
                        print("Processed", count)
            
# with open("venv\Lib\site-packages\poke_env\data\static\moves\gen9moves.json") as f:
#     data = json.load(f)

#     for key in data.keys():
#         if(re.search("([^a-z A-Z])", key)):
#             print(key)
import BattleUtilities
from poke_env.player.player import Player
from poke_env.environment import AbstractBattle
from config import api_key
from openai import OpenAI
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import SideCondition
from difflib import SequenceMatcher
import os
import json
import time

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def findSimilar(fullText, search, requiredRatio=0.9):
    startPointer = 0
    endPointer = len(search) 
    while(endPointer < len(fullText)):
        comparisonText = fullText[startPointer:endPointer]
        if(similar(comparisonText, search) >= requiredRatio):
            return True
        
        startPointer+=1
        endPointer+=1


    return False

typeMatchups = '''
The type matchups are:
,,Normal,Fighting,Flying,Poison,Ground,Rock,Bug,Ghost,Steel,Fire,Water,Grass,Electric,Psychic,Ice,Dragon,Dark,Fairy
,Normal,1x,1x,1x,1x,1x,0.5x,1x,0x,0.5x,1x,1x,1x,1x,1x,1x,1x,1x,1x
,Fighting,2x,1x,0.5x,0.5x,1x,2x,0.5x,0x,2x,1x,1x,1x,1x,0.5x,2x,1x,2x,0.5x
,Flying,1x,2x,1x,1x,1x,0.5x,2x,1x,0.5x,1x,1x,2x,0.5x,1x,1x,1x,1x,1x
,Poison,1x,1x,1x,0.5x,0.5x,0.5x,1x,0.5x,0x,1x,1x,2x,1x,1x,1x,1x,1x,2x
,Ground,1x,1x,0x,2x,1x,2x,0.5x,1x,2x,2x,1x,0.5x,2x,1x,1x,1x,1x,1x
,Rock,1x,0.5x,2x,1x,0.5x,1x,2x,1x,0.5x,2x,1x,1x,1x,1x,2x,1x,1x,1x
,Bug,1x,0.5x,0.5x,0.5x,1x,1x,1x,0.5x,0.5x,0.5x,1x,2x,1x,2x,1x,1x,2x,0.5x
,Ghost,0x,1x,1x,1x,1x,1x,1x,2x,1x,1x,1x,1x,1x,2x,1x,1x,0.5x,1x
,Steel,1x,1x,1x,1x,1x,2x,1x,1x,0.5x,0.5x,0.5x,1x,0.5x,1x,2x,1x,1x,2x
,Fire,1x,1x,1x,1x,1x,0.5x,2x,1x,2x,0.5x,0.5x,2x,1x,1x,2x,0.5x,1x,1x
,Water,1x,1x,1x,1x,2x,2x,1x,1x,1x,2x,0.5x,0.5x,1x,1x,1x,0.5x,1x,1x
,Grass,1x,1x,0.5x,0.5x,2x,2x,0.5x,1x,0.5x,0.5x,2x,0.5x,1x,1x,1x,0.5x,1x,1x
,Electric,1x,1x,2x,1x,0x,1x,1x,1x,1x,1x,2x,0.5x,0.5x,1x,1x,0.5x,1x,1x
,Psychic,1x,2x,1x,2x,1x,1x,1x,1x,0.5x,1x,1x,1x,1x,0.5x,1x,1x,0x,1x
,Ice,1x,1x,2x,1x,2x,1x,1x,1x,0.5x,0.5x,0.5x,2x,1x,1x,0.5x,2x,1x,1x
,Dragon,1x,1x,1x,1x,1x,1x,1x,1x,0.5x,1x,1x,1x,1x,1x,1x,2x,1x,0x
,Dark,1x,0.5x,1x,1x,1x,1x,1x,2x,1x,1x,1x,1x,1x,2x,1x,1x,0.5x,0.5x
,Fairy,1x,2x,1x,0.5x,1x,1x,1x,1x,0.5x,0.5x,1x,1x,1x,1x,1x,2x,2x,1x

where the attacking type is shown in the rows with the defending type in the columns
For Pokemon of multiple types, effectiveness is multiplicatively added (for example 2x and 2x is 4x and 0.5x and 2x is 1x)
'''
class LLMPlayer(Player): 

    def __init__(self, battle_format, usingAssistant=True):
        super().__init__(battle_format=battle_format)

        self.client = OpenAI(
        api_key=api_key,
        )
        self.tried = False

        with open("venv\Lib\site-packages\poke_env\data\static\moves\gen8moves.json") as f:
            self.moveData = json.load(f)

        with open("venv\Lib\site-packages\poke_env\data\static\pokedex\gen8pokedex.json") as f:
            self.pokemonData = json.load(f)
        
        self.usingAssistant = usingAssistant

        # Assistants require getting the asssistant and making a thread
        if(usingAssistant):
            self.assistant = self.client.beta.assistants.retrieve( assistant_id="asst_1vXPZEKgnSDRaesCCXkFiFom")
            
        

    def convertPartyPokemonToPrompt(self, identifier: str, pokemon: Pokemon):
        promptText = ""
        promptText += " ".join(identifier.split(" ")[1:]) + " "
        promptText += str(pokemon.current_hp) + "/" + str(pokemon.max_hp) + " hp (fraction of total)\n"
        promptText += "type(s): " + pokemon.type_1.name.lower() + " "
        if(pokemon.type_2):
            promptText += pokemon.type_2.name.lower() + "\n"
        else:
            promptText += "\n"

        promptText += "statuses:\n"
        if(pokemon.status):
            if(pokemon.status.name == "FNT"):
                return ""
            promptText += pokemon.status.name + "\n"
        else:
            promptText += "None\n"
        
        if(pokemon.ability):
            promptText += "ability: " + pokemon.ability + "\n"
        else:
            promptText += "ability: unknown\n"

        if(pokemon.gender):
            promptText += "gender: " + pokemon.gender.name + "\n"
        else:
            promptText += "gender: unknown\n"

        if(pokemon.item):
            promptText += "item: " + pokemon.item + "\n"
        else:
            promptText += "item: unknown\n"

        # Maybe we'll build this
        # if(pokemon.revealed):
        #     promptText += "Revealed\n"
        # else:
        #     promptText += "Hidden\n"
        if(len(pokemon.moves.items()) > 0):
            promptText += "Moves: "
            for identifier, move in pokemon.moves.items():
                promptText += move.id + " ( power: " +str(move.base_power) + ", accuracy: " + str(move.accuracy) + ", type: " + move.type.name.lower() +", priority: "+str(move.priority) + ", category: " +move.category.name + " )\n"
        
        promptText += "\n"
        return promptText
    
    def convertActivePokemonToPrompt(self, identifier: str, pokemon: Pokemon):
        promptText= " ".join(identifier.split(" ")[1:])  + "\n"
        foundStatBoots = False
        if(pokemon.species in self.pokemonData):
            promptText += "Its base stats are: \n" 
            for statName, stat in self.pokemonData[pokemon.species]["baseStats"].items():
                promptText += statName +": " + str(stat) + "\n"
        promptText += "It has the following stat boosts:\n"
        for boost, amount in pokemon.boosts.items():
            if(amount != 0):
                promptText += boost + " " + str(amount) + "\n"
                foundStatBoots = True
        if(foundStatBoots == False):
            promptText += "None\n"
        else:
            promptText += "Where boosts go from -6 to 6 and disappear on switching\n"
        if(pokemon.first_turn):
            promptText += "It's this pokemon's first turn out\n"
        
        for effect, counter in pokemon.effects.items():
            promptText += "It is affected by " + effect.name.lower() + "\n"
        
        return promptText
    
    def getMoveFromChatCompletion(self, chat_completion, battle):
        gptParagraphs = chat_completion.choices[0].message.content.split("\n")
        #gptChoice = gptParagraphs[-1].lower().replace(" ", "").replace("-", "")
        for gptChoice in reversed(gptParagraphs):
            gptChoice = gptChoice.lower().replace(" ", "").replace("-", "")
            for move in battle.available_moves:
                if(findSimilar(gptChoice, move.id)):
                    print("Found move:", move.id)
                    return self.create_order(move)
            
            switchIndex = gptChoice.find("switch")
            if(switchIndex != -1):
                for pokemon in battle.available_switches:
                    if("baseSpecies" in self.pokemonData[pokemon.species]):
                        pokemonSearchName = self.pokemonData[pokemon.species]["baseSpecies"].lower().replace(" ", "").replace("-", "")
                    elif (pokemon.species in self.pokemonData):
                        pokemonSearchName = self.pokemonData[pokemon.species]["name"].lower().replace(" ", "").replace("-", "")
                    else:
                        pokemonSearchName = pokemon.species.lower().replace(" ", "").replace("-", "")
                    if(findSimilar(gptChoice[switchIndex:], pokemonSearchName)):
                        print("Found switch pokemon:", pokemon.species)
                        return self.create_order(pokemon)
                    
            print("No action found in paragraph\n")
            
        print("No action found in response")
        return self.choose_random_move(battle)
    
    def getMoveFromJSON(self, jsonObject, battle : AbstractBattle):
        actionObject = jsonObject

        if(actionObject["action_type"] == "move"):
            moveChoice = actionObject["move"].lower().replace(" ", "").replace("-", "") + " "
            for move in battle.available_moves:
                #print(moveChoice, move.id)
                if(findSimilar(moveChoice, move.id)):
                    print("Found move:", move.id)
                    # Safety for avoiding dynamaxing when you can't
                    if(not battle.can_dynamax and not battle.active_pokemon.is_dynamaxed):
                        return self.create_order(move, dynamax=False)
                    else:
                        return self.create_order(move, dynamax=actionObject["dynamax"])
        else:
            switchChoice = actionObject["pokemon"].lower().replace(" ", "").replace("-", "") + " "
            for pokemon in battle.available_switches:
                if("baseSpecies" in self.pokemonData[pokemon.species]):
                    pokemonSearchName = self.pokemonData[pokemon.species]["baseSpecies"].lower().replace(" ", "").replace("-", "")
                elif (pokemon.species in self.pokemonData):
                    pokemonSearchName = self.pokemonData[pokemon.species]["name"].lower().replace(" ", "").replace("-", "")
                else:
                    pokemonSearchName = pokemon.species.lower().replace(" ", "").replace("-", "")
                
                #print(switchChoice, pokemonSearchName)
                if(findSimilar(switchChoice, pokemonSearchName)):
                    print("Found switch pokemon:", pokemon.species)
                    return self.create_order(pokemon)
                
        print("no actions found")
        return self.choose_random_move(battle)


    def choose_move(self, battle):
        prompt = "You are playing turn " + str(battle.turn) + " of a generation 8 random battle.  Your team is:\n"

        activePokemon = None
        for identifier, pokemon in battle.team.items():
            if(pokemon.active):
                activePokemon = identifier, pokemon

            prompt += self.convertPartyPokemonToPrompt(identifier, pokemon)

        prompt += "Your opponent's team is: \n"
        opponentActivePokemon = None
        for identifier, pokemon in battle.opponent_team.items():
            if(pokemon.active):
                opponentActivePokemon = identifier, pokemon

            prompt += self.convertPartyPokemonToPrompt(identifier, pokemon)
        
        remainingPokemon = 6 - len(battle.opponent_team.items())
        prompt += "and " + str(remainingPokemon) + " unseen pokemon\n"

        for weather, turn in battle.weather.items():
            prompt += "There is " + weather.name + " that started on turn " + str(turn) + "\n"

        for field, turn in battle.fields.items():
            prompt += "There is " + field.name + " that started on turn " + str(turn) + "\n"      

        for condition, stacks in battle.side_conditions.items():
            if(condition == SideCondition.SPIKES or condition == SideCondition.TOXIC_SPIKES):
                prompt += "There are " + str(stacks) + " stacks of " + condition.name.lower() + " on your side\n"
            else:
                prompt += "Your side is already affected by " + condition.name.lower() + "\n"

        for condition, stacks in battle.opponent_side_conditions.items():
            if(condition == SideCondition.SPIKES or condition == SideCondition.TOXIC_SPIKES):
                prompt += "There are " + str(stacks) + " stacks of " + condition.name.lower() + " on affecting the opponent's side\n"
            else:
                prompt += "Their side is already affected by " + condition.name.lower() + " which doesn't stack\n"

        
        
        prompt += "Your active pokemon is " + self.convertActivePokemonToPrompt(activePokemon[0], activePokemon[1])
        prompt += "Your oppoent's active pokemon is " + self.convertActivePokemonToPrompt(opponentActivePokemon[0], opponentActivePokemon[1])

        if(not self.usingAssistant):
            prompt += typeMatchups
            prompt += "Evaluate which types are super effective against the opposing pokemon and your own active pokemon and then choose the best action from the following \n"

        prompt += "Choose the best action from the following \n"

        #prompt += str(battle.available_moves)
        if(battle.available_moves):
            for move in battle.available_moves:
                prompt += self.moveData[move.id]["name"] + "\n"

        if(battle.available_switches):
            for switch in battle.available_switches:
                prompt += "switch to " +str(switch.species) + "\n"
        prompt += "\n"

        # If not using Absol assistant, needs this
        if(not self.usingAssistant):
            prompt += "Respond in a JSON of an \"explanation\" field with a string of an evaluation of the actions to take, whether it is a move or switch in the field \"action_type\" with the string \"move\" representing a move and \"switch\" representing a switch, the move to use or pokemon to switch to in fields \"move\" and \"pokemon\", and then whether or not to dynamax as a boolean"

        if(not self.tried and len(battle.available_moves) > 0):
            #print(prompt)
            #self.tried = True
            if(self.usingAssistant):
                #Use assistant
                # Make new thread to avoid rate limit
                self.thread = self.client.beta.threads.create()
                message = self.client.beta.threads.messages.create(
                    thread_id=self.thread.id,
                    role="user",
                    content=prompt
                )
                
                run = self.client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                )
                waitCount = 0
                time.sleep(3)
                while(True):
                    run = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=run.id
                    )
                    if(run.status == "completed"):
                        break
                    else:
                        print(run.status)
                        if(run.status != "in_progress" and run.status != "queued"):
                            print("erroring out, doing random for rest of run")
                            self.tried = True
                            return self.choose_random_move(battle)
                        waitCount += 1
                        if(waitCount > 50):
                            print("Timeout!")
                            try:
                                self.client.beta.threads.runs.cancel(thread_id=self.thread.id, run_id=run.id)
                                time.sleep(2)
                            except:
                                print("problem cancelling, likely just completed")
                            return self.choose_random_move(battle)

                        #print("Waiting...")
                        time.sleep(2)

                messages = self.client.beta.threads.messages.list(
                    thread_id=self.thread.id
                    )
                if(not os.path.exists("prompts/" + battle._battle_tag + "/") ):
                    os.mkdir("prompts/" + battle._battle_tag + "/")

                with open("prompts/" + battle._battle_tag + "/" + battle.player_username + "_" + str(battle._turn) + ".txt", "w+", encoding='utf-8') as f:
                    f.write(prompt + "\n" + messages.data[0].content[0].text.value)

                
                rawString = messages.data[0].content[0].text.value
                try:
                    startIndex = rawString.index("```json")
                    endIndex = rawString[startIndex + 7:].index("```")
                    data = json.loads(rawString[startIndex + 7: startIndex + 7 + endIndex])
                except:
                    try:
                        data = json.loads(rawString)
                    except:
                        print("no json detected")
                        return self.choose_random_move(battle)

                return self.getMoveFromJSON(jsonObject=data, battle=battle)
            else:
                #Using default chatgpt
                try:
                    chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                        ],
                        response_format={ "type": "json_object" },
                        model="gpt-4-1106-preview",
                        timeout=20.0
                    )
                except:
                    print("api timeout")
                    return self.choose_random_move(battle)

                if(not os.path.exists("prompts/" + battle._battle_tag + "/") ):
                    os.mkdir("prompts/" + battle._battle_tag + "/")

                with open("prompts/" + battle._battle_tag + "/" + battle.player_username + "_" + str(battle._turn) + ".txt", "w+") as f:
                    f.write(prompt + "\n" + chat_completion.choices[0].message.content)

            return self.getMoveFromJSON(chat_completion=json.loads(chat_completion.choices[0].message.content), battle=battle)
        
        return self.choose_random_move(battle)

                    
            


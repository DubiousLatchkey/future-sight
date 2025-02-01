from openai import OpenAI
from config import api_key
import time
import json
import re

client = OpenAI(
        api_key=api_key,
        )
assistant = client.beta.assistants.retrieve( assistant_id="asst_1vXPZEKgnSDRaesCCXkFiFom")
thread = client.beta.threads.create()
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="""You are playing turn 29 of a generation 8 random battle. Your team is: Mr. Mime 89/209 hp type(s): psychic fairy statuses: None ability: filter gender: MALE item: choicescarf Moves: healingwish ( power: 0, accuracy: 1.0, type: psychic, priority: 0, category: STATUS ) psychic ( power: 90, accuracy: 1.0, type: psychic, priority: 0, category: SPECIAL ) focusblast ( power: 120, accuracy: 0.7, type: fighting, priority: 0, category: SPECIAL ) dazzlinggleam ( power: 80, accuracy: 1.0, type: fairy, priority: 0, category: SPECIAL )

Weezing 252/252 hp type(s): poison fairy statuses: None ability: levitate gender: MALE item: blacksludge Moves: sludgebomb ( power: 90, accuracy: 1.0, type: poison, priority: 0, category: SPECIAL ) painsplit ( power: 0, accuracy: 1.0, type: normal, priority: 0, category: STATUS ) fireblast ( power: 110, accuracy: 0.85, type: fire, priority: 0, category: SPECIAL ) strangesteam ( power: 90, accuracy: 0.95, type: fairy, priority: 0, category: SPECIAL )

Marowak 14/238 hp type(s): fire ghost statuses: None ability: lightningrod gender: MALE item: thickclub Moves: poltergeist ( power: 110, accuracy: 0.9, type: ghost, priority: 0, category: PHYSICAL ) flamecharge ( power: 50, accuracy: 1.0, type: fire, priority: 0, category: PHYSICAL ) flareblitz ( power: 120, accuracy: 1.0, type: fire, priority: 0, category: PHYSICAL ) earthquake ( power: 100, accuracy: 1.0, type: ground, priority: 0, category: PHYSICAL )

Flygon 88/259 hp type(s): ground dragon statuses: None ability: levitate gender: FEMALE item: leftovers Moves: outrage ( power: 120, accuracy: 1.0, type: dragon, priority: 0, category: PHYSICAL ) uturn ( power: 70, accuracy: 1.0, type: bug, priority: 0, category: PHYSICAL ) defog ( power: 0, accuracy: 1.0, type: flying, priority: 0, category: STATUS ) earthquake ( power: 100, accuracy: 1.0, type: ground, priority: 0, category: PHYSICAL )

Your opponent's team is: Bronzong 24/100 hp type(s): steel psychic statuses: BRN ability: levitate gender: NEUTRAL item: leftovers Moves: ironhead ( power: 80, accuracy: 1.0, type: steel, priority: 0, category: PHYSICAL )

and 0 unseen pokemon Your active pokemon is Flygon Its base stats are: atk: 100 def: 80 hp: 80 spa: 80 spd: 80 spe: 100 It has the following stat boosts: None It's this pokemon's first turn out Your oppoent's active pokemon is Bronzong Its base stats are: atk: 89 def: 116 hp: 67 spa: 79 spd: 116 spe: 33 It has the following stat boosts: None It's this pokemon's first turn out

Choose the best action from the following: Outrage U-turn Defog Earthquake switch to weezinggalar switch to mrmime switch to marowakalola"""
)

run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
)
while(True):
    run = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
    )
    if(run.status == "completed"):
        break
    else:
        print("Waiting...")
        time.sleep(1)

messages = client.beta.threads.messages.list(
  thread_id=thread.id
)

#print(messages)
print(messages.data[0].content[0].text.value)

rawString = messages.data[0].content[0].text.value
startIndex = rawString.index("```json")
endIndex = rawString[startIndex + 7:].index("```")
data = json.loads(rawString[startIndex + 7: startIndex + 7 + endIndex])

print(data)
import torch
import numpy as np
import torch.nn as nn
import json
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance
from torch.utils.data import DataLoader, Dataset
import pickle
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.battle import AbstractBattle
from tqdm import trange

from makePokemonEmbedding import Pokemon_SkipGram_Model
from makeMoveEmbedding import Move_SkipGram_Model
from poke_env.player.player import Player
import re
from poke_env.environment import PokemonGender, Status, Weather, SideCondition, Field

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class ActionPredictionDataset(Dataset):
    def __init__(self, transform=None, useFile=False):
        self.transform = transform

        self.data = []

        if(useFile):
            with open("pokemonBattleData.pickle", "rb") as f:
                self.data = pickle.load(f)
        else:

            for player in os.listdir("trainingReplays/"):
                for replay in os.listdir("trainingReplays/" + player):
                    with open("trainingReplays/" + player + "/" + replay) as f:
                        text = "\n".join([i for i in f.read().split("\n") if len(i) > 0 and i[0] == "|"])
                        # check if player 1 or 2
                        if(re.search("p1|" + player, text)):
                            otherPlayerID = "p2"
                        else:
                            otherPlayerID = "p1"

                        # check if won and not a loss, draw, or inconclusive
                        #print(text.split("\n")[-1])
                        if("faint|" + otherPlayerID in text.split("\n")[-1]):
                            #print("player won")
                            # This player won the battle, add all game state data to dataset
                            battleID = replay.split(" - ")[-1][:-5]
                            for filename in os.listdir("gameStates/" + battleID):
                                if(player in filename):
                                    with open("gameStates/" + battleID + "/" + filename) as f:
                                        gameStateData = json.load(f)
                                        self.data.append(gameStateData)
                        #print("player didn't win")

                print(player, len(self.data))
            with open("pokemonBattleData.pickle", "wb") as f:
                pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

class ActionPredictionModel(nn.Module):

    def __init__(self):
        super(ActionPredictionModel, self).__init__()
        

        self.fc1 = nn.Linear(
            in_features=8244,
            out_features= 5000
        )
        self.fc2 = nn.Linear(
            in_features=5000,
            out_features= 1000
        )
        self.fc3 = nn.Linear(
            in_features=1000,
            out_features=130
        )

    def forward(self, inputs_ ):
        x = self.fc1(inputs_)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
    

class NNPlayer(Player): 

    def __init__(self, battle_format):
        super().__init__(battle_format=battle_format)

        self.model = ActionPredictionModel()

        with open("pokemonToId.pickle", "rb") as f:
            # Currently missing alcremie forms, probably because poke_env registers species name as cosmetic form first
            self.pokemonToId = pickle.load(f)
            #print(len(self.pokemonToId), self.pokemonToId)
        
        self.idToPokemon = {v: k for k, v in self.pokemonToId.items()}

        pokemonEmbeddingModel = Pokemon_SkipGram_Model(vocab_size=len(self.pokemonToId.keys())).to(device)
        pokemonEmbeddingModel.load_state_dict(torch.load("pokemonModel.pt"))
        self.pokemonEmbeddingModel = pokemonEmbeddingModel.linear.weight.cpu().detach().numpy()

        self.pokemonDistMatrix = self.get_distance_matrix(self.pokemonEmbeddingModel, 'cosine')

        self.moveToIndex = {}
        with open("venv\Lib\site-packages\poke_env\data\static\moves\gen9moves.json", "r") as f:
            data = json.load(f)
            count = 1
            for i in data:
                self.moveToIndex[i] = count
                count += 1
            self.moveToIndex["<unk>"] = 0
        self.indexToMove = {v: k for k, v in self.moveToIndex.items()}

        moveEmbeddingModel = Move_SkipGram_Model(len(self.moveToIndex.keys())).to(device)
        moveEmbeddingModel.load_state_dict(torch.load("moveModel.pt"))
        self.moveEmbeddingModel = moveEmbeddingModel.linear.weight.cpu().detach()

        self.moveDistMatrix = self.get_distance_matrix(self.moveEmbeddingModel, 'cosine')

        torch.manual_seed(42)

        self.statusEmbedding = nn.Embedding(
            num_embeddings=8,
            embedding_dim=4,
            max_norm=1,
        )
        
        self.genderEmbedding = nn.Embedding(
            num_embeddings=4,
            embedding_dim=2,
            max_norm=1
        )

        # Item embedding
        self.itemToIndex = {}
        if(not os.path.exists("itemDict.pickle")):
            count = 0
            for foldername in os.listdir("gameStates/"):
                with open(os.path.join("gameStates/", foldername, "MinimaxPlayer 1_1.json") ) as f:
                    data = json.load(f)
                    pokemons = data["team"] + data["opponent_team"]
                    for pokemon in pokemons:
                        if(pokemon["item"] not in self.itemToIndex):
                            self.itemToIndex[pokemon["item"]] = count
                            count += 1
                with open(os.path.join("gameStates/", foldername, "MinimaxPlayer 2_1.json") ) as f:
                    data = json.load(f)
                    pokemons = data["team"] + data["opponent_team"]
                    for pokemon in pokemons:
                        if(pokemon["item"] not in self.itemToIndex):
                            self.itemToIndex[pokemon["item"]] = count
                            count += 1
            with open("itemDict.pickle", "wb") as f:
                pickle.dump(self.itemToIndex, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        else:
            with open("itemDict.pickle", "rb") as f:
                self.itemToIndex = pickle.load(f)
        #print(self.itemToIndex, len(self.itemToIndex.items()))

        self.itemEmbedding = nn.Embedding(
            num_embeddings=len(self.itemToIndex.items()),
            embedding_dim=32,
            max_norm=1
        )

        # Weather Embedding
        self.weatherEmbedding = nn.Embedding(
            num_embeddings=10,
            embedding_dim=4,
            max_norm=1
        )

        # Field Embedding
        self.fieldEmbedding = nn.Embedding(
            num_embeddings=14,
            embedding_dim=8,
            max_norm=1
        )

        # Side Conditions embedding
        self.sideConditionEmbedding = nn.Embedding(
            num_embeddings=21,
            embedding_dim=8,
            max_norm=1
        )

        self.switchesInARow = 0

        

    def processPokemon(self, pokemon: Pokemon):
        # Pokemon name, moves, active, status, gender, boosts
        if(pokemon):
            pokemonData = torch.from_numpy(self.pokemonEmbeddingModel[self.pokemonToId.get(pokemon.species, 0)])
            moves = sorted([i[1] for i in pokemon.moves.items()], key= lambda x: x.id)
            move1 = self.moveEmbeddingModel[self.moveToIndex[moves[0].id]] if len(moves) > 0 else self.moveEmbeddingModel[self.moveToIndex["<unk>"]]
            move2 = self.moveEmbeddingModel[self.moveToIndex[moves[1].id]] if len(moves) > 1 else self.moveEmbeddingModel[self.moveToIndex["<unk>"]]
            move3 = self.moveEmbeddingModel[self.moveToIndex[moves[2].id]] if len(moves) > 2 else self.moveEmbeddingModel[self.moveToIndex["<unk>"]]
            move4 = self.moveEmbeddingModel[self.moveToIndex[moves[3].id]] if len(moves) > 3 else self.moveEmbeddingModel[self.moveToIndex["<unk>"]]

            # convert moves to tensors from numpy
            # move1 = torch.from_numpy(move1)
            # move2 = torch.from_numpy(move2)
            # move3 = torch.from_numpy(move3)
            # move4 = torch.from_numpy(move4)


            active = torch.tensor([pokemon.active], dtype=torch.int16)

            status = self.statusEmbedding(torch.tensor(pokemon.status.value, dtype=torch.long)) if pokemon.status else self.statusEmbedding(torch.tensor(7, dtype=torch.long))

            gender = self.genderEmbedding(torch.tensor(pokemon.gender.value, dtype=torch.long)) if pokemon.gender else self.statusEmbedding(torch.tensor(3, dtype=torch.long))

            atkBoost = torch.tensor([pokemon.boosts["atk"]])
            defBoost = torch.tensor([pokemon.boosts["def"]])
            spaBoost = torch.tensor([pokemon.boosts["spa"]])
            spdBoost = torch.tensor([pokemon.boosts["spd"]])
            speBoost = torch.tensor([pokemon.boosts["spe"]])
            accuracyBoost = torch.tensor([pokemon.boosts["accuracy"]])
            evasionBoost = torch.tensor([pokemon.boosts["evasion"]])

            hp = torch.tensor([pokemon.current_hp_fraction])

            item = self.itemEmbedding(torch.tensor(self.itemToIndex[pokemon.item], dtype=torch.long)) if pokemon.item else self.itemEmbedding(torch.tensor(self.itemToIndex["unknown_item"], dtype=torch.long))

            #print(pokemonData.shape, hp.shape, move1.shape, move2.shape, move3.shape, move4.shape, active.shape, status.shape, gender.shape, atkBoost.shape, defBoost.shape, spaBoost.shape, spdBoost.shape, speBoost.shape, accuracyBoost.shape, evasionBoost.shape, hp.shape)
        

        else:
            pokemonData = torch.from_numpy(self.pokemonEmbeddingModel[self.pokemonToId["<unk>"]])
            move1 = self.moveEmbeddingModel[self.moveToIndex["<unk>"]]
            move2 = self.moveEmbeddingModel[self.moveToIndex["<unk>"]]
            move3 = self.moveEmbeddingModel[self.moveToIndex["<unk>"]]
            move4 = self.moveEmbeddingModel[self.moveToIndex["<unk>"]]
            active = torch.tensor([False], dtype=torch.int16)

            status = self.statusEmbedding(torch.tensor(7, dtype=torch.long))

            gender = self.genderEmbedding(torch.tensor(3, dtype=torch.long))

            atkBoost = torch.tensor([0])
            defBoost = torch.tensor([0])
            spaBoost = torch.tensor([0])
            spdBoost = torch.tensor([0])
            speBoost = torch.tensor([0])
            accuracyBoost = torch.tensor([0])
            evasionBoost = torch.tensor([0])

            hp = torch.tensor([1.0])

            item = self.itemEmbedding(torch.tensor(self.itemToIndex["unknown_item"], dtype=torch.long))

            #print(pokemonData.shape, hp.shape, move1.shape, move2.shape, move3.shape, move4.shape, active.shape, status.shape, gender.shape, atkBoost.shape, defBoost.shape, spaBoost.shape, spdBoost.shape, speBoost.shape, accuracyBoost.shape, evasionBoost.shape, hp.shape)
        
        return torch.concatenate((pokemonData, hp, move1, move2, move3, move4, active, status, gender, atkBoost, defBoost, spaBoost, spdBoost, speBoost, accuracyBoost, evasionBoost, item), dim=0)

    def processPokemonDict(self, pokemon):
        if(pokemon):
            pokemonData = torch.from_numpy(self.pokemonEmbeddingModel[self.pokemonToId.get(pokemon.species, 0)])
            moves = sorted([i[1] for i in pokemon["moves"]])
            move1 = self.moveEmbeddingModel[self.moveToIndex[moves[0].id]] if len(moves) > 0 else self.moveEmbeddingModel[self.moveToIndex["<unk>"]]
            move2 = self.moveEmbeddingModel[self.moveToIndex[moves[1].id]] if len(moves) > 1 else self.moveEmbeddingModel[self.moveToIndex["<unk>"]]
            move3 = self.moveEmbeddingModel[self.moveToIndex[moves[2].id]] if len(moves) > 2 else self.moveEmbeddingModel[self.moveToIndex["<unk>"]]
            move4 = self.moveEmbeddingModel[self.moveToIndex[moves[3].id]] if len(moves) > 3 else self.moveEmbeddingModel[self.moveToIndex["<unk>"]]

            # convert moves to tensors from numpy
            # move1 = torch.from_numpy(move1)
            # move2 = torch.from_numpy(move2)
            # move3 = torch.from_numpy(move3)
            # move4 = torch.from_numpy(move4)


            active = torch.tensor([pokemon.active], dtype=torch.int16)

            status = self.statusEmbedding(torch.tensor(Status[pokemon["status"]].value, dtype=torch.long)) if pokemon["status"] != "null" else self.statusEmbedding(torch.tensor(7, dtype=torch.long))

            gender = self.genderEmbedding(torch.tensor(PokemonGender[pokemon["gender"]].value, dtype=torch.long)) if pokemon["gender"] != "null" else self.statusEmbedding(torch.tensor(3, dtype=torch.long))

            accuracyBoost = torch.tensor([pokemon["boosts"][0][1]])
            atkBoost = torch.tensor([pokemon["boosts"][1][1]])
            defBoost = torch.tensor([pokemon["boosts"][2][1]])
            evasionBoost = torch.tensor([pokemon["boosts"][3][1]])
            spaBoost = torch.tensor([pokemon["boosts"][4][1]])
            spdBoost = torch.tensor([pokemon["boosts"][5][1]])
            speBoost = torch.tensor([pokemon["boosts"][6][1]])

            hp = torch.tensor([pokemon["current_hp"] / pokemon["max_hp"]])

            item = self.itemEmbedding(torch.tensor(self.itemToIndex[pokemon["item"]], dtype=torch.long)) if pokemon["item"] else self.itemEmbedding(torch.tensor(self.itemToIndex["unknown_item"], dtype=torch.long))

            #print(pokemonData.shape, hp.shape, move1.shape, move2.shape, move3.shape, move4.shape, active.shape, status.shape, gender.shape, atkBoost.shape, defBoost.shape, spaBoost.shape, spdBoost.shape, speBoost.shape, accuracyBoost.shape, evasionBoost.shape, hp.shape)
        

        else:
            pokemonData = torch.from_numpy(self.pokemonEmbeddingModel[self.pokemonToId["<unk>"]])
            move1 = self.moveEmbeddingModel[self.moveToIndex["<unk>"]]
            move2 = self.moveEmbeddingModel[self.moveToIndex["<unk>"]]
            move3 = self.moveEmbeddingModel[self.moveToIndex["<unk>"]]
            move4 = self.moveEmbeddingModel[self.moveToIndex["<unk>"]]
            active = torch.tensor([False], dtype=torch.int16)

            status = self.statusEmbedding(torch.tensor(7, dtype=torch.long))

            gender = self.genderEmbedding(torch.tensor(3, dtype=torch.long))

            atkBoost = torch.tensor([0])
            defBoost = torch.tensor([0])
            spaBoost = torch.tensor([0])
            spdBoost = torch.tensor([0])
            speBoost = torch.tensor([0])
            accuracyBoost = torch.tensor([0])
            evasionBoost = torch.tensor([0])

            hp = torch.tensor([1.0])

            item = self.itemEmbedding(torch.tensor(self.itemToIndex["unknown_item"], dtype=torch.long))

            #print(pokemonData.shape, hp.shape, move1.shape, move2.shape, move3.shape, move4.shape, active.shape, status.shape, gender.shape, atkBoost.shape, defBoost.shape, spaBoost.shape, spdBoost.shape, speBoost.shape, accuracyBoost.shape, evasionBoost.shape, hp.shape)
        
        return torch.concatenate((pokemonData, hp, move1, move2, move3, move4, active, status, gender, atkBoost, defBoost, spaBoost, spdBoost, speBoost, accuracyBoost, evasionBoost, item))

    def processGameStateIntoMLFormat(self, gameState):
        # Embed your team
        myPokemon = sorted([i[1] for i in gameState["team"]], key= lambda x: x["species"])
        pokemon1 = self.processPokemonDict(myPokemon[0]) if len(myPokemon) > 0 else self.processPokemonDict(None)
        pokemon2 = self.processPokemonDict(myPokemon[1]) if len(myPokemon) > 1 else self.processPokemonDict(None)
        pokemon3 = self.processPokemonDict(myPokemon[2]) if len(myPokemon) > 2 else self.processPokemonDict(None)
        pokemon4 = self.processPokemonDict(myPokemon[3]) if len(myPokemon) > 3 else self.processPokemonDict(None)
        pokemon5 = self.processPokemonDict(myPokemon[4]) if len(myPokemon) > 4 else self.processPokemonDict(None)
        pokemon6 = self.processPokemonDict(myPokemon[5]) if len(myPokemon) > 5 else self.processPokemonDict(None)

        # Embed opponent's team
        theirPokemon = sorted([i[1] for i in gameState["opponent_team"]], key= lambda x: x["species"])
        opponentPokemon1 = self.processPokemonDict(theirPokemon[0]) if len(theirPokemon) > 0 else self.processPokemonDict(None)
        opponentPokemon2 = self.processPokemonDict(theirPokemon[1]) if len(theirPokemon) > 1 else self.processPokemonDict(None)
        opponentPokemon3 = self.processPokemonDict(theirPokemon[2]) if len(theirPokemon) > 2 else self.processPokemonDict(None)
        opponentPokemon4 = self.processPokemonDict(theirPokemon[3]) if len(theirPokemon) > 3 else self.processPokemonDict(None)
        opponentPokemon5 = self.processPokemonDict(theirPokemon[4]) if len(theirPokemon) > 4 else self.processPokemonDict(None)
        opponentPokemon6 = self.processPokemonDict(theirPokemon[5]) if len(theirPokemon) > 5 else self.processPokemonDict(None)

        # Weather, Fields, Side Conditions
        # TODO: add support for more than one field / condition
        weather = self.weatherEmbedding(Weather[gameState["weather"][0]].value) if len(gameState["weather"]) > 0 else self.weatherEmbedding(torch.tensor(9, dtype=torch.long))

        field = self.fieldEmbedding(Field[gameState["fields"][0]].value) if len(gameState["fields"]) > 0 else self.fieldEmbedding(torch.tensor(13, dtype=torch.long))

        sideCondition = self.sideConditionEmbedding(SideCondition[gameState["side_conditions"][0]].value) if len(gameState["side_conditions"]) > 0 else self.sideConditionEmbedding(torch.tensor(20, dtype=torch.long))

        opponentSideCondition = self.sideConditionEmbedding(SideCondition[gameState["opponent_side_conditions"][0]].value) if len(gameState["opponent_side_conditions"]) > 0 else self.sideConditionEmbedding(torch.tensor(20, dtype=torch.long))

        x = torch.concatenate((pokemon1, pokemon2, pokemon3, pokemon4, pokemon5, pokemon6, opponentPokemon1, opponentPokemon2, opponentPokemon3, opponentPokemon4, opponentPokemon5, opponentPokemon6, weather, field, sideCondition, opponentSideCondition))
        
        # get actual move chosen
        terrastallize = torch.tensor(gameState["terrastallize"])
        moveOrSwitch = torch.tensor(1) if gameState["action_type"] == "switch" else torch.tensor(0)
        if(moveOrSwitch):
            # it's a switch, encode pokemon
            actionData = torch.from_numpy(self.pokemonEmbeddingModel[self.pokemonToId.get(gameState["action"], 0)])
        else:
            # it's a move, encode move 
            actionData = self.moveEmbeddingModel[self.moveToIndex[gameState["action"]]]

        y = torch.concatenate((moveOrSwitch, terrastallize, actionData))

        return x
    
    def runInference(self, battle: AbstractBattle):
        # Embed your team
        myPokemon = sorted([i[1] for i in battle.team.items()], key= lambda x: x.species)
        pokemon1 = self.processPokemon(myPokemon[0]) if len(myPokemon) > 0 else self.processPokemon(None)
        pokemon2 = self.processPokemon(myPokemon[1]) if len(myPokemon) > 1 else self.processPokemon(None)
        pokemon3 = self.processPokemon(myPokemon[2]) if len(myPokemon) > 2 else self.processPokemon(None)
        pokemon4 = self.processPokemon(myPokemon[3]) if len(myPokemon) > 3 else self.processPokemon(None)
        pokemon5 = self.processPokemon(myPokemon[4]) if len(myPokemon) > 4 else self.processPokemon(None)
        pokemon6 = self.processPokemon(myPokemon[5]) if len(myPokemon) > 5 else self.processPokemon(None)

        # Embed opponent's team
        theirPokemon = sorted([i[1] for i in battle.opponent_team.items()], key= lambda x: x.species)
        opponentPokemon1 = self.processPokemon(theirPokemon[0]) if len(theirPokemon) > 0 else self.processPokemon(None)
        opponentPokemon2 = self.processPokemon(theirPokemon[1]) if len(theirPokemon) > 1 else self.processPokemon(None)
        opponentPokemon3 = self.processPokemon(theirPokemon[2]) if len(theirPokemon) > 2 else self.processPokemon(None)
        opponentPokemon4 = self.processPokemon(theirPokemon[3]) if len(theirPokemon) > 3 else self.processPokemon(None)
        opponentPokemon5 = self.processPokemon(theirPokemon[4]) if len(theirPokemon) > 4 else self.processPokemon(None)
        opponentPokemon6 = self.processPokemon(theirPokemon[5]) if len(theirPokemon) > 5 else self.processPokemon(None)

        #print(pokemon1.shape, opponentPokemon1.shape, opponentPokemon2.shape)
        # Concat with Conditions
        x = torch.concatenate((pokemon1, pokemon2, pokemon3, pokemon4, pokemon5, pokemon6, opponentPokemon1, opponentPokemon2, opponentPokemon3, opponentPokemon4, opponentPokemon5, opponentPokemon6))

        #print(x.shape)
        x = self.model(x)

        return x
    
    def get_distance_matrix(self, wordvecs, metric):
        dist_matrix = distance.squareform(distance.pdist(wordvecs, metric))
        return dist_matrix

    def evaluateNNOutput(self, outputTensor, correctAnswer=None, battle : AbstractBattle = None):
        #with open("outputTensorTest.pickle", "wb") as f:
        #    pickle.dump(outputTensor, f, protocol=pickle.HIGHEST_PROTOCOL)

        moveOrSwitch = int(outputTensor[0] > 1 - outputTensor[0] )
        if(type(outputTensor) == type(torch.zeros(1))):
            actionNPArray = outputTensor[2:].detach().numpy()
        else:
            actionNPArray = np.array(outputTensor[2:])
        if(correctAnswer):
            if( moveOrSwitch != correctAnswer[0]):
                # you chose poorly
                return False
            
        terrasallize = int(outputTensor[1] > 1 - outputTensor[1] )
        if(correctAnswer and terrasallize != correctAnswer[1]):
            return False
        else:
            #check pokemon or move
            if(moveOrSwitch == 0):
                # get all move options and appropriate embeddings
                cosineDistances = []
                #for embedding in self.moveEmbeddingModel:
                    # compare output to embeddings
                    #cosineDistances.append(distance.cdist(torch.tensor([outputTensor[2:]]), torch.tensor([embedding]), 'cosine'))
                cosineDistances = distance.cdist(torch.tensor(np.array([actionNPArray])), self.moveEmbeddingModel, 'cosine')

                if(not correctAnswer):
                    # There is no correct answer, find the indices of the moves of the pokemon and just argmin those

                    movesPokemonKnows = []
                    moveNamesPokemonKnows = []
                    for move in battle.active_pokemon.moves.values():
                        moveIndex = self.moveToIndex.get(move.id, 0)
                        movesPokemonKnows.append(cosineDistances[0][moveIndex])
                        moveNamesPokemonKnows.append(move.id)

                    moveChosenIndex = np.argmin(np.array(movesPokemonKnows))
                    moveChosenName = moveNamesPokemonKnows[moveChosenIndex]
                    #print(moveChosenName)
                    return moveOrSwitch, terrasallize, moveChosenName

                # Get closest move, see if it matches actual chosen
                moveChosenIndex = np.argmin(np.array(cosineDistances))
                correctMoveIndex = self.moveEmbeddingModel.index(correctAnswer[2:])

                # check move
                if(moveChosenIndex == correctMoveIndex):
                    return True
                else:
                    return False
                
            else:
                # It's a switch, check pokemon instead
                cosineDistances = []
                #for embedding in self.pokemonEmbeddingModel:
                    # compare output to embeddings
                    #cosineDistances.append(distance.cdist(torch.tensor([outputTensor[2:]]), torch.tensor([embedding]), 'cosine'))
                cosineDistances = distance.cdist(torch.tensor(np.array([actionNPArray])), self.pokemonEmbeddingModel, 'cosine')
                
                # Get closest move, see if it matches actual chosen
                pokemonChosenIndex = np.argmin(np.array(cosineDistances))

                if(not correctAnswer):
                    availablePokemon = []
                    availablePokemonNames = []
                    for pokemon in battle.team.values():
                        if(not pokemon.active):
                            pokemonIndex = self.pokemonToId.get(pokemon.species, 0)
                            availablePokemon.append(cosineDistances[0][pokemonIndex])
                            availablePokemonNames.append(pokemon.species)

                    pokemonChosenIndex = np.argmin(np.array(availablePokemon))
                    pokemonChosenName = availablePokemonNames[pokemonChosenIndex]
                    #pokemonChosenName = self.idToPokemon[pokemonChosenIndex]
                    #print(pokemonChosenName)
                    return moveOrSwitch, terrasallize, pokemonChosenName

                correctPokemonIndex = self.pokemonEmbeddingModel.index(correctAnswer[2:])

                # check move
                if(pokemonChosenIndex == correctPokemonIndex):
                    return True
                else:
                    return False
                            
    
    def train(training_dataloader, validation_dataloader, model, loss_function, optimizer, epochs):
        validation_losses = []
        validation_accuracies = []
        training_losses = []
        training_accuracies = []
        
        for i in range(epochs):
            size = len(training_dataloader.dataset)
            pbar = trange(size, unit="carrots")
            pbar.set_description(f"Epoch: {i}")
            batch_training_loss = 0
            correct = 0
            for batch, (x, y) in enumerate(training_dataloader):        
                x = x.cuda()
                y = y.cuda()
                pred = model(x)
                loss = loss_function(pred, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_training_loss += loss
                
                #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                pbar.set_postfix(loss=loss.item())
            pbar.close()
                
            training_losses.append(batch_training_loss.item() / size)
            #training_accuracies.append(correct / size)
                
            # # Validate
            # size = len(validation_dataloader.dataset)
            # validation_loss, validation_correct = 0, 0

            # with torch.no_grad():
            #     for x, y in validation_dataloader:
            #         x = x.cuda()
            #         y = y.cuda()
            #         pred = model(x)
            #         validation_loss += loss_function(pred, y).item()
            #         validation_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    
            # validation_loss /= size
            # validation_correct /= size
            # print(f"Epoch: {i+1} Validation Error: \n Accuracy: {(100*validation_correct):>0.1f}%, Avg loss: {validation_loss:>8f} \n")
            # validation_losses.append(validation_loss)
            # validation_accuracies.append(validation_correct)
    
        return model, training_losses, validation_losses, training_accuracies, validation_accuracies
        

    def choose_move(self, battle):
        
        y = self.runInference(battle)
        #print(y)
        #print(y.shape)

        # y[0] is move or switch
        # y[1] is terrastallize
        # y[2:130] is move/pokemon

        moveOrSwitch, terrastalize, actionName = self.evaluateNNOutput(y, battle=battle)

        # Safeguard against reaching turn limit from 2 dumb ais
        if(self.switchesInARow >= 30):
            return self.choose_random_move(battle)

        if(moveOrSwitch == 0):
            # It's a move, create order
            self.switchesInARow = 0
            for move in battle.available_moves:
                if(move.id == actionName):
                    #print("move: ", move.id)
                    if(not battle.can_tera):
                        return self.create_order(order=move, terastallize=False)
                    else:
                        return self.create_order(order=move, terastallize=terrastalize)
        else:
            # It's a switch, create order
            self.switchesInARow += 1
            for pokemon in battle.available_switches:
                if(pokemon.species == actionName):
                    #print("switch:", pokemon.id)
                    return self.create_order(order=pokemon)

        # No move or pokemon found
        return self.choose_random_move(battle)
    

if(__name__ == "__main__"):
    # print("loading dataset")
    # dataset = ActionPredictionDataset(transform=processGameStateIntoMLFormat, useFile=True)
    # print("done loading")

    NNPlayer = NNPlayer(battle_format="gen9randombattle")

    testEmbed = nn.Embedding(2, 5)
    print([i for i in testEmbed.parameters()])
    print(testEmbed(torch.tensor(1)), testEmbed(torch.tensor(1)).shape, testEmbed(torch.tensor(1)).view((1, -1)))
    print(PokemonGender["MALE"].value)
import numpy as np
from poke_env.player import Gen9EnvSinglePlayer, Player
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gymnasium.spaces import Space, Box
from gensim.models import Word2Vec
from poke_env.data import GenData, to_id_str
from poke_env.environment import PokemonGender, Status, Weather, SideCondition, Field
from poke_env.environment import AbstractBattle
from random import randint

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class BattleModel(nn.Module):
    def __init__(self,
                 bench_feature_dim: int,   
                 num_bench: int,           
                 global_feature_dim: int,  # e.g., features like status conditions, weather, etc.
                 conv_channels: int = 64,  # number of output channels for the convolution
                 lstm_hidden_size: int = 256,
                 lstm_layers: int = 2,
                 output_size: int = 102      # example: output could be action logits or value predictions
                 ):
        super(BattleModel, self).__init__()

        # Convolution over bench Pokémon.
        # We treat the bench as a sequence where each “time step” is one Pokémon,
        # and each Pokémon is represented by bench_feature_dim channels.
        # nn.Conv1d expects input of shape (batch, channels, sequence_length).
        self.bench_conv = nn.Conv1d(in_channels=bench_feature_dim,
                                    out_channels=conv_channels,
                                    kernel_size=1)  

        # Use an adaptive pooling layer to collapse the bench dimension.
        # This will yield one vector per sample, irrespective of how many bench Pokémon there are.
        self.pool = nn.AdaptiveAvgPool1d(1)  # output shape: (batch, conv_channels, 1)

        # After pooling, we have conv_channels features from the bench.
        # These will be concatenated with the global battle features.
        lstm_input_size = conv_channels + global_feature_dim

        # LSTM to process the combined state.
        # We assume here that each observation is one time step.
        # If you process a sequence of observations, you can feed in the whole sequence.
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True)

        # Final output layer.
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, bench: torch.Tensor, global_features: torch.Tensor, hidden=None):
        """
        Parameters:
          bench: Tensor of shape (batch_size, num_bench, bench_feature_dim)
          global_features: Tensor of shape (batch_size, global_feature_dim)
          hidden: Optional hidden state for the LSTM

        Returns:
          output: The final output (e.g., action logits or value)
          hidden: Updated LSTM hidden state
        """
        bench = bench.unsqueeze(1)
        # Permute bench to match nn.Conv1d input: (batch, channels, sequence_length)
        bench = bench.permute(0, 2, 1)  # now shape is (batch_size, bench_feature_dim, extra for cnn)

        # Apply convolution and a ReLU nonlinearity.
        bench_conv = F.relu(self.bench_conv(bench))  # shape: (batch_size, conv_channels, extra for cnn)

        # Pool across the bench dimension to get one vector per sample.
        bench_pooled = self.pool(bench_conv)  # shape: (batch_size, conv_channels, 1)
        bench_pooled = bench_pooled.squeeze(-1)  # shape: (batch_size, conv_channels)

        # Concatenate the bench representation with global battle features.
        # global_features is assumed to be of shape (batch_size, global_feature_dim)
        combined = torch.cat([bench_pooled, global_features], dim=1)  # shape: (batch_size, conv_channels + global_feature_dim)

        # If this is a single time step observation, add a time dimension for the LSTM.
        # LSTM expects (batch_size, seq_len, input_size). Here, seq_len = 1.
        combined = combined.unsqueeze(1)  # shape: (batch_size, 1, lstm_input_size)

        # Pass the combined representation through the LSTM.
        lstm_out, hidden = self.lstm(combined, hidden)  # lstm_out: (batch_size, 1, lstm_hidden_size)

        # Remove the time dimension.
        lstm_out = lstm_out.squeeze(1)  # shape: (batch_size, lstm_hidden_size)

        # Compute the final output.
        output = self.fc(lstm_out)  # shape: (batch_size, output_size)
        return output, hidden
            

class Agent:
    def __init__(self):
        self.model = BattleModel(bench_feature_dim=5000, num_bench=5, global_feature_dim=1190, output_size=102).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def update_policy(self, rewards, similarities):
        # Calculate discounted returns
        returns = []
        G = 0
        gamma = 0.96
        # Compute returns in reverse order for efficiency.
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        similarities = torch.tensor(similarities).requires_grad_(True)

        policy_loss = []
        for similarity, G in zip(similarities, returns):
            # If using a baseline, subtract it from G to get the advantage.
            policy_loss.append(-similarity * G)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        print(f"Similarity: {similarities.mean()}, Reward: {sum(rewards)}, Loss: {loss}")


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def visualize(self):
        input_names = ['Bench Species data', 'Battle Data']
        output_names = ['Action']
        exampleIn1 = torch.tensor([0.0 for i in range(5000)]).unsqueeze(0).to(device)
        exampleIn2 = torch.tensor([0.0 for i in range(1190)]).unsqueeze(0).to(device)
        torch.onnx.export(self.model, args=(exampleIn1, exampleIn2) , f='model.onnx', input_names=input_names, output_names=output_names)

    

class LSTMPlayer(Player):
    def __init__(self, agent, playerString="p1a"):
        super().__init__(save_replays=True)
        # load in the model
        self.agent = agent

        # load in embedding
        self.embeddingModel = Word2Vec.load("skipgram_model.model")

        self.lastObservation = None

        self.switchesInARow = 0

        self.reward = 0
        self.similarities = []
        self.averageSimilarity = 0

        self.rewards = []

        self.playerString = playerString

    def popRewardsAndSimilarities(self):
        rewards = self.rewards
        similarities = self.similarities
        self.similarities = []
        self.rewards = []
        return rewards, similarities

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )
        
    def _battle_finished_callback(self, battle: AbstractBattle):
        # Calculate reward for game
        reward = 0

        # Count fainted enemy pokemon
        enemyFainted = 0
        for name, pokemon in battle.opponent_team.items():
            if(pokemon.fainted):
                enemyFainted += 1

        # Count fainted my pokemon
        myFainted = 0
        for name, pokemon in battle.team.items():
            if(pokemon.fainted):
                myFainted += 1

        netFainted = enemyFainted - myFainted
        reward += netFainted * 2
        
        # Check if you won
        if(battle.won > 0):
            reward += 30

        # Save reward
        self.reward = reward
        self.rewards.append(reward)

        # Calculate average similarity
        self.averageSimilarity = torch.mean(torch.tensor(self.similarities)).requires_grad_(True)
        #self.similarities = []

    def _handle_battle_message(self, split_messages):

        reward = 0
        if(len(split_messages) > 2):
            # Loop through and mark rewards
            myAction = True
            for split_message in split_messages[1:]:

                if(len(split_message) < 2):
                    continue
                
                #print(split_message, end=" ")
                if(split_message[1] == "move"):
                    # Assign who performed the move
                    if(self.playerString in split_message[2]):
                        myAction = True
                    else:
                        myAction = False

                elif(split_message[1] == "-damage"):
                    # Maybe reward damage?
                    pass
                elif(split_message[1] == "-resisted"):
                    # Reward if you resisted, punish if was resisted
                    reward += 0.5 if not myAction else -0.5
                    #print("resisted")
                elif(split_message[1] == "-immune"):
                    # Reward if immune, punish if ineffective
                    reward += 0.75 if not myAction else -0.75
                    #print("immune")
                elif(split_message[1] == "-supereffective"):
                    # Reward if you supered, punish if was supered
                    reward += 0.5 if myAction else -0.5
                    #print("super")


        self.rewards.append(reward)
            
        return super()._handle_battle_message(split_messages)

    
    def embed_team(self, team):
        # Embed your team
        # Find active pokemon
        found = False
        for name, pokemon in team.items():
            if pokemon.active:
                activePokemonName, activePokemon = name, pokemon
                found = True
                break
        if(found):
            team.pop(activePokemonName)
        else:
            # When forced to make a switch, there is no active pokemon
            if(len(list(team)) > 0):
                activePokemonName, activePokemon = list(team.items())[0]
                team.pop(activePokemonName) 
            else:
                # No knowledge on pokemon yet
                activePokemon = None

        team = list(team.items())

        # Embed non active pokemon
        bench_embeddings = []

        # Loop over the bench positions.
        for i in range(5):
            # If there is a Pokémon at this bench index, use its data.
            if i < len(team):
                pkmn_id = to_id_str(team[i][1].name)
                pkmn = team[i][1]
                # Get the Pokémon's own embedding.
                pkmn_emb = np.array(self.embeddingModel.wv[pkmn_id])
                
                # For moves, iterate up to max_moves.
                move_embs = []
                moves = list(pkmn.moves.items())
                for j in range(4):
                    if j < len(moves):
                        move_token = moves[j][1].id
                        move_emb = np.array(self.embeddingModel.wv[move_token])
                    else:
                        move_emb = np.array(self.embeddingModel.wv["<unk>"])
                    move_embs.append(move_emb)
                
                # Concatenate the Pokémon embedding with its move embeddings.
                # Each embedding is assumed to be 100 dimensions.
                # The resulting vector will have 100 + 4*100 = 500 dimensions.
                move_embs = np.concatenate(move_embs)
                bench_pkmn_emb = np.concatenate([pkmn_emb, move_embs])
            else:
                # If no Pokémon exists at this bench slot, fill with <unk> embeddings.
                unk_emb = np.array(self.embeddingModel.wv["<unk>"])
                bench_pkmn_emb = np.concatenate((unk_emb, unk_emb, unk_emb, unk_emb, unk_emb))
            
            bench_embeddings.append(bench_pkmn_emb)
    
    
        bench = np.concatenate(bench_embeddings)

        # Embed active pokemon and team features
        if(activePokemon):
            activePokemonTensor = np.array(self.embeddingModel.wv[to_id_str(activePokemon.name)])
            activePokemonMoves = list(activePokemon.moves.items())
        else:
            activePokemonTensor = np.array(self.embeddingModel.wv["<unk>"])
            activePokemonMoves = []

        move_embs = []
        for i in range(4):
            if i < len(activePokemonMoves):
                move_token = activePokemonMoves[i][1].id
                move_emb = np.array(self.embeddingModel.wv[move_token])
            else:
                move_emb = np.array(self.embeddingModel.wv["<unk>"])
            move_embs.append(move_emb)
        move_embs = np.concatenate(move_embs)
        hps = np.array([pokemon.current_hp_fraction for name, pokemon in team] + [1] * (6 - len(team)))
        atkBoosts = np.array([pokemon.boosts["atk"] for name, pokemon in team] + [0] * (6 - len(team)))
        spaBoosts = np.array([pokemon.boosts["spa"] for name, pokemon in team] + [0] * (6 - len(team)))
        defBoosts = np.array([pokemon.boosts["def"] for name, pokemon in team] + [0] * (6 - len(team)))
        spdBoosts = np.array([pokemon.boosts["spd"] for name, pokemon in team] + [0] * (6 - len(team)))
        speBoosts = np.array([pokemon.boosts["spe"] for name, pokemon in team] + [0] * (6 - len(team)))
        accuracyBoosts = np.array([pokemon.boosts["accuracy"] for name, pokemon in team] + [0] * (6 - len(team)))
        evasionBoosts = np.array([pokemon.boosts["evasion"] for name, pokemon in team] + [0] * (6 - len(team)))
        genders = np.array([pokemon.gender.value if pokemon.gender else 3 for name, pokemon in team] + [3] * (6 - len(team)))
        statuses = np.array([pokemon.status.value if pokemon.status else 7 for name, pokemon in team] + [7] * (6 - len(team)))

        teamFeatures = np.concatenate([activePokemonTensor, move_embs, hps, atkBoosts, spaBoosts, defBoosts, spdBoosts, speBoosts, accuracyBoosts, evasionBoosts, genders, statuses])
        return bench, teamFeatures
    
    def embed_battle(self, battle) :

        # Embed teams
        myBench, myTeamFeatures = self.embed_team(battle.team)
        opponentBench, opponentTeamFeatures = self.embed_team(battle.opponent_team)

        # Embed global features
        weather_features = []
        for weather in Weather:
            if weather in battle.weather:
                weather_features.append(battle.weather[weather] - battle.turn)
            else:
                weather_features.append(0)
        weather_features = np.array(weather_features)

        field_features = []
        for field in Field:
            if field in battle.fields:
                field_features.append(battle.fields[field] - battle.turn)
            else:
                field_features.append(0)
        field_features = np.array(field_features)

        sideCondition_features = []
        for sideCondition in SideCondition:
            if sideCondition in battle.side_conditions:
                sideCondition_features.append(battle.side_conditions[sideCondition] - battle.turn)
            else:
                sideCondition_features.append(0)
        sideCondition_features = np.array(sideCondition_features)

        opponentSideCondition_features = []
        for sideCondition in SideCondition:
            if sideCondition in battle.opponent_side_conditions:
                opponentSideCondition_features.append(battle.opponent_side_conditions[sideCondition] - battle.turn)
            else:
                opponentSideCondition_features.append(0)
        opponentSideCondition_features = np.array(opponentSideCondition_features)

        global_features = np.concatenate([weather_features, field_features, sideCondition_features, opponentSideCondition_features])

        return np.concatenate([myBench, opponentBench], dtype=np.float32), np.concatenate([myTeamFeatures, opponentTeamFeatures, global_features], dtype=np.float32)


    def choose_move(self, battle : AbstractBattle):

        if(self.switchesInARow >= 30):
            return self.choose_random_move(battle)

        #print("calling choose move")
        # Embed the battle state
        battleStateBench, battleState = self.embed_battle(battle)
        #self.lastObservation = battleState, battleStateBench
        
        # Process inputs
        battleStateBench = torch.tensor(battleStateBench).unsqueeze(0).to(device)
        battleState = torch.tensor(battleState).unsqueeze(0).to(device)

        #print("battle embeded")
        # Run the model
        logits, _ = self.agent.model(battleStateBench, battleState)
        #print("model run")
        logits = logits[0]
        # Determine actions based on logits
        switch = logits[1] > 0.5
        if(not switch):
            tera = logits[0] > 0.5
        else:
            tera = False
        
        query_vector = logits[2:].cpu().detach().numpy()
        

        if switch:
            self.switchesInARow += 1
            #print(battle.available_switches)
            switches = [(self.embeddingModel.wv[to_id_str(pokemon.name())], pokemon) for pokemon in battle.available_switches]
            similarities = [(pokemon, 1 - cosine(pokemonVector, query_vector)) for pokemonVector, pokemon in switches]
            if(len(similarities) > 0):
                closest_pokemon = max(similarities, key=lambda x: x[1])[0]

                # Calculate similarity to final performed switch for weighting rewards later
                pokemonVectorPlusTeraSwitch = torch.tensor(self.embeddingModel.wv[to_id_str(closest_pokemon.name())])
                pokemonVectorPlusTeraSwitch = torch.cat((pokemonVectorPlusTeraSwitch, torch.tensor([0, 1]))).to(device)
                self.similarities.append( F.cosine_similarity(logits, pokemonVectorPlusTeraSwitch, dim=0) )

                return self.create_order(order=closest_pokemon)
            else:
                print("no switches available, going random")
                return self.choose_random_move(battle)
        else:
            self.switchesInARow = 0
            #print(battle.available_moves)
            moves = [(move, self.embeddingModel.wv[move.id] if move.id in self.embeddingModel.wv.key_to_index else np.zeros(100)) for move in battle.available_moves]
            similarities = [(move, 1 - cosine(moveVector, query_vector)) for move, moveVector in moves]
            if(len(similarities) > 0):
                closest_move = max(similarities, key=lambda x: x[1])[0]

                # Calculate similarity to final performed move for weighting rewards later
                moveVectorPlusTeraSwitch = torch.tensor(self.embeddingModel.wv[closest_move.id] if closest_move.id in self.embeddingModel.wv.key_to_index else np.zeros(100))
                teraInt = 1 if tera and battle.can_tera else 0
                moveVectorPlusTeraSwitch = torch.cat((moveVectorPlusTeraSwitch, torch.tensor([teraInt, 0]))).to(device)
                self.similarities.append( F.cosine_similarity(logits, moveVectorPlusTeraSwitch, dim=0) )

                return self.create_order(order=closest_move, terastallize=tera and battle.can_tera)
            else:
                #print("no moves available, going random")
                return self.choose_random_move(battle)
        

    def describe_embedding(self) -> Space:
        #low = [-1 for i in range(5000)] + [-1 for i in range(500)] + [0 for i in range(6)] + [0 for i in range(42)] + [0 for i in range(6)] + [0 for i in range(6)] + [-1 for i in range(500)] + [0 for i in range(6)] + [0 for i in range(42)] + [0 for i in range(6)] + [0 for i in range(6)] + [-10 for i in range(70)]
        #high = [1 for i in range(5000)] + [1 for i in range(500)] + [1000 for i in range(6)] + [6 for i in range(42)] + [3 for i in range(6)] + [7 for i in range(6)] + [1 for i in range(500)] + [1000 for i in range(6)] + [6 for i in range(42)] + [3 for i in range(6)] + [7 for i in range(6)] + [0 for i in range(70)]
        low = [-1000 for i in range(6190)]
        high = [1000 for i in range(6190)]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
    def reset(self, *, seed = None, options = None):
        
        super().reset(seed=seed)
        if(self.current_battle):
            self.lastObservation = self.embed_battle(self.current_battle)
            #print(self.observation_space)
            #print(len(self.lastObservation[0]), len(self.lastObservation[1]))
            #print(np.concatenate(self.lastObservation).dtype)
            return np.concatenate(self.lastObservation), {}
        else:
            return np.array([0 for i in range(6190)]), {}

import numpy as np
from poke_env.player import Gen9EnvSinglePlayer
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
import torch.nn.functional as F
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
                 lstm_layers: int = 1,
                 output_size: int = 10      # example: output could be action logits or value predictions
                 ):
        super(BattleModel, self).__init__()

        # Convolution over bench Pokémon.
        # We treat the bench as a sequence where each “time step” is one Pokémon,
        # and each Pokémon is represented by bench_feature_dim channels.
        # nn.Conv1d expects input of shape (batch, channels, sequence_length).
        self.bench_conv = nn.Conv1d(in_channels=bench_feature_dim,
                                    out_channels=conv_channels,
                                    kernel_size=1)  # kernel_size=1 is like a per-Pokémon transform

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
        # Permute bench to match nn.Conv1d input: (batch, channels, sequence_length)
        bench = bench.permute(0, 2, 1)  # now shape is (batch_size, bench_feature_dim, num_bench)

        # Apply convolution and a ReLU nonlinearity.
        bench_conv = F.relu(self.bench_conv(bench))  # shape: (batch_size, conv_channels, num_bench)

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
    

class LSTMPlayer(Gen9EnvSinglePlayer):
    def __init__(self, opponent):
        super().__init__(opponent=opponent, start_challenging=True, save_replays=True)
        # load in the model
        self.model = BattleModel(bench_feature_dim=5000, num_bench=5, global_feature_dim=1132, output_size=102)

        # load in embedding
        self.embeddingModel = Word2Vec.load("skipgram_model.model")

        self.lastObservation = None

        self.switchesInARow = 0

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )
    
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

        print("calling choose move")
        # Embed the battle state
        battleStateBench, battleState = self.embed_battle(battle)
        self.lastObservation = battleState, battleStateBench

        print("battle embeded")
        # Run the model
        logits, _ = self.model((torch.tenrsor(battleStateBench), torch.tensor(battleState)))
        print("model run")
        # Determine actions based on logits
        switch = logits[1] > 0.5
        if(not switch):
            tera = logits[0] > 0.5
        else:
            tera = False
        
        query_vector = logits[2:]
        
        if switch:
            self.switchesInARow += 1
            switches = [(self.embeddingModel(to_id_str(pokemon.name())), pokemon) for pokemon in battle.available_switches()]
            similarities = [(pokemon, 1 - cosine(pokemonVector, query_vector)) for pokemonVector, pokemon in switches]
            closest_pokemon = max(similarities, key=lambda x: x[1])[0]

            return self.create_order(order=closest_pokemon)
        else:
            self.switchesInARow = 0
            moves = [(move, self.embeddingModel(move.id)) for move in battle.available_moves()]
            similarities = [(move, 1 - cosine(moveVector)) for move, moveVector in moves]
            closest_move = max(similarities, key=lambda x: x[1])[0]

            return self.create_order(order=closest_move, terastallize=tera and battle.can_tera)
        

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

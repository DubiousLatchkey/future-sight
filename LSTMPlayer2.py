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

    def visualize(self):
        input_names = ['Bench Species data', 'Battle Data']
        output_names = ['Action']
        exampleIn1 = torch.tensor([0.0 for i in range(5000)]).unsqueeze(0).to(device)
        exampleIn2 = torch.tensor([0.0 for i in range(1190)]).unsqueeze(0).to(device)
        torch.onnx.export(self, args=(exampleIn1, exampleIn2) , f='model.onnx', input_names=input_names, output_names=output_names)

            


class ActorCriticBattleModel(nn.Module):
    def __init__(self,
                 bench_feature_dim: int,   # dimensionality of bench features
                 global_feature_dim: int,  # dimensionality of global battle features
                 hidden_dim: int = 1024,    # hidden dimension for the FC layers
                 policy_output_size: int = 100,  # number of actions for the policy head
                 switching_output_size: int = 1,  # e.g., binary decision: switch or not
                 terastallize_output_size: int = 1  # e.g., binary decision: terastallize or not
                 ):
        super(ActorCriticBattleModel, self).__init__()

        # The input is a concatenation of bench features and global features.
        input_dim = bench_feature_dim + global_feature_dim

        # Two shared fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor head: outputs logits for policy
        self.policy_head = nn.Linear(hidden_dim, policy_output_size)

        # Critic head: outputs a scalar value
        self.critic_head = nn.Linear(hidden_dim, 1)

        # Additional heads for switching and terastallizing decisions
        self.switching_head = nn.Linear(hidden_dim, switching_output_size)
        self.terastallize_head = nn.Linear(hidden_dim, terastallize_output_size)

    def forward(self, bench: torch.Tensor, global_features: torch.Tensor):
        """
        Parameters:
          bench: Tensor of shape (batch_size, bench_feature_dim)
          global_features: Tensor of shape (batch_size, global_feature_dim)
          
        Returns:
          policy: Softmax probabilities over actions, shape (batch_size, policy_output_size)
          critic: Value estimate, shape (batch_size, 1)
          switching: Logits for the switching decision, shape (batch_size, switching_output_size)
          terastallize: Logits for the terastallize decision, shape (batch_size, terastallize_output_size)
        """
        # Concatenate bench and global features
        x = torch.cat([bench, global_features], dim=1)

        # Pass through shared fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)

        # Compute outputs
        policy_logits = self.policy_head(x)
        policy = F.softmax(policy_logits, dim=-1)  # Convert logits to probability distribution

        critic = self.critic_head(x)  # State value estimation

        switching = torch.sigmoid(self.switching_head(x))  # Logits for switching decision
        terastallize = torch.sigmoid(self.terastallize_head(x))  # Logits for terastallization decision

        return policy, critic, switching, terastallize

    def visualize(self):
        input_names = ['Bench Species data', 'Battle Data']
        output_names = ['Action', 'critic', 'switching', 'tera']
        exampleIn1 = torch.tensor([0.0 for i in range(5000)]).unsqueeze(0).to(device)
        exampleIn2 = torch.tensor([0.0 for i in range(1121)]).unsqueeze(0).to(device)
        torch.onnx.export(self, args=(exampleIn1, exampleIn2) , f='model.onnx', input_names=input_names, output_names=output_names)


torch.autograd.set_detect_anomaly(True)

class Agent:
    def __init__(self):
        self.embeddingModel = Word2Vec.load("skipgram_model.model")
        self.model = ActorCriticBattleModel(bench_feature_dim=5000, global_feature_dim=1121, policy_output_size=len(self.embeddingModel.wv)).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.policy_losses = []
        self.critic_losses = []
        self.switching_losses = []
        self.terastallize_losses = []

    def pop_losses(self):
        losses = {
            "policy_losses": self.policy_losses.copy(),
            "critic_losses": self.critic_losses.copy(),
            "switching_losses": self.switching_losses.copy(),
            "tera_losses": self.terastallize_losses.copy()
        }
        self.policy_losses.clear()
        self.critic_losses.clear()
        self.switching_losses.clear()
        self.terastallize_losses.clear()
        return losses

    def update_policy(self, rewards, log_probs, values, switching_losses, terastallize_losses, retain_graph=False):
        # Calculate discounted returns
        
        R = rewards[-1]
        returns = [R]
        gamma = 0.95
        # Compute returns in reverse order for efficiency.
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns[:-1])
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_losses = []
        critic_losses = []


        # Compute losses for each step
        for log_prob, value, R in zip(log_probs, values, returns):
            # Advantage: how much better the outcome was compared to the critic's estimate
            advantage = R - value.item()
            
            # Policy loss (using the REINFORCE trick with baseline)
            policy_losses.append(-log_prob * advantage)
            
            # Critic loss: mean squared error between estimated value and the observed return
            # (Here we wrap R in a tensor to match dimensions.)
            critic_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(device)))

        # Sum the per-step losses (you could also take a mean, depending on your setup)
        policy_loss = torch.stack(policy_losses).sum()
        critic_loss = torch.stack(critic_losses).sum()
        if abs(critic_loss) > 20:
            critic_loss = torch.tensor(20.0 if critic_loss > 0 else -20.0)
        switching_loss = torch.stack(switching_losses).sum()
        terastallize_loss = torch.stack(terastallize_losses).sum()

        loss = ( 2 * policy_loss + 0.1 * critic_loss + 0.5 * switching_loss + 0.5 * terastallize_loss ) / 4

        # Backpropagation: zero gradients, compute gradients, and update parameters.
        self.optimizer.zero_grad()  # optimizer assumed to be defined (e.g., Adam(model.parameters(), lr=...))
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()

        print(f"Critic Loss: {critic_loss}, Reward: {sum(rewards)}, Policy Loss: {policy_loss}, Loss: {loss}")
        #print(returns)
        self.policy_losses.append(policy_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.switching_losses.append(switching_loss.item())
        self.terastallize_losses.append(terastallize_loss.item())



    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


    

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

        self.logProbabilities = []
        self.criticValues = []
        self.teraLosses = []
        self.switchingLosses = []

        self.rewards = []

        self.playerString = playerString

    def popTrainingVariables(self):
        rewards = self.rewards.copy()
        critics = self.criticValues.copy()
        teraLosses = self.teraLosses.copy()
        switchingLosses = self.switchingLosses.copy()
        logProbabilities = self.logProbabilities.copy()
        self.logProbabilities.clear()
        self.criticValues.clear()
        self.switchingLosses.clear()
        self.teraLosses.clear()
        self.rewards.clear()
        return rewards, logProbabilities, critics, switchingLosses, teraLosses

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
        turn = 1
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
                elif(split_message[1] == "turn"):
                    turn = int(split_message[2])

        # Keep the games short bro
        reward -= turn * 0.01


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
            atkBoost = np.array([activePokemon.boosts["atk"]])
            spaBoost = np.array([activePokemon.boosts["spa"]])
            defBoost = np.array([activePokemon.boosts["def"]])
            spdBoost = np.array([activePokemon.boosts["spd"]])
            speBoost = np.array([activePokemon.boosts["spe"]])
            accuracyBoost = np.array([activePokemon.boosts["accuracy"]])
            evasionBoost = np.array([activePokemon.boosts["evasion"]])
        else:
            activePokemonTensor = np.array(self.embeddingModel.wv["<unk>"])
            activePokemonMoves = []
            atkBoost = np.zeros(1)
            spaBoost = np.zeros(1)
            defBoost = np.zeros(1)
            spdBoost = np.zeros(1)
            speBoost = np.zeros(1)
            accuracyBoost = np.zeros(1)
            evasionBoost = np.zeros(1)

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
        
        genders = np.array([pokemon.gender.value if pokemon.gender else 3 for name, pokemon in team] + [3] * (6 - len(team)))
        statuses = np.array([pokemon.status.value if pokemon.status else 7 for name, pokemon in team] + [7] * (6 - len(team)))

        teamFeatures = np.concatenate([activePokemonTensor, move_embs, hps, atkBoost, spaBoost, defBoost, spdBoost, speBoost, accuracyBoost, evasionBoost, genders, statuses])
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

        global_features = np.concatenate([weather_features, field_features, sideCondition_features, opponentSideCondition_features, [battle.turn]])

        return np.concatenate([myBench, opponentBench], dtype=np.float32), np.concatenate([myTeamFeatures, opponentTeamFeatures, global_features], dtype=np.float32)


    def choose_move(self, battle : AbstractBattle):

        if(self.switchesInARow >= 30):
            self.switchesInARow = 0
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
        policy, critic, switching, terastallize = self.agent.model(battleStateBench, battleState)
        #print("model run")
        # Determine actions based on logits
        switch = 1.0 if switching > 0.5 else 0.0
        if(not switch):
            tera = 1.0 if terastallize > 0.5 and battle.can_tera else 0.0
        else:
            tera = 0.0

        # Critic value and losses added
        self.criticValues.append(critic.squeeze())
        switchingLoss = F.mse_loss(switching.squeeze(), torch.tensor(switch).to(device))
        teraLoss = F.mse_loss(terastallize.squeeze(), torch.tensor(tera).to(device))
        self.switchingLosses.append(switchingLoss)
        self.teraLosses.append(teraLoss)

        #Choose new move based on these new things
        policySqueezed = policy.squeeze()

        if switch:
            self.switchesInARow += 1
            switches = [(policySqueezed[self.embeddingModel.wv.get_index(to_id_str(pokemon.name), len(self.embeddingModel.wv) - 1)], pokemon) for pokemon in battle.available_switches]
            if len(switches) < 1: return self.choose_random_move(battle)
            probability, closest_pokemon = max(switches, key=lambda x: x[0])
            self.logProbabilities.append(probability)

            return self.create_order(order=closest_pokemon)
        else:
            self.switchesInARow = 0
            moves = [(policySqueezed[self.embeddingModel.wv.get_index(move.id, default=len(self.embeddingModel.wv) - 1)], move) for move in battle.available_moves]
            if len(moves) < 1: return self.choose_random_move(battle)
            probability, closest_move = max(moves, key=lambda x: x[0])
            self.logProbabilities.append(probability)

            return self.create_order(order=closest_move, terastallize=bool(tera) and battle.can_tera)
        
        #query_vector = policy.cpu().detach().numpy()
        

        # if switch:
        #     self.switchesInARow += 1
        #     #print(battle.available_switches)
        #     switches = [(self.embeddingModel.wv[to_id_str(pokemon.name())], pokemon) for pokemon in battle.available_switches]
        #     similarities = [(pokemon, 1 - cosine(pokemonVector, query_vector)) for pokemonVector, pokemon in switches]
        #     if(len(similarities) > 0):
        #         closest_pokemon = max(similarities, key=lambda x: x[1])[0]

        #         # Calculate similarity to final performed switch for weighting rewards later
        #         pokemonVectorPlusTeraSwitch = torch.tensor(self.embeddingModel.wv[to_id_str(closest_pokemon.name())])
        #         pokemonVectorPlusTeraSwitch = torch.cat((pokemonVectorPlusTeraSwitch, torch.tensor([0, 1]))).to(device)
        #         self.similarities.append( F.cosine_similarity(logits, pokemonVectorPlusTeraSwitch, dim=0) )

        #         return self.create_order(order=closest_pokemon)
        #     else:
        #         print("no switches available, going random")
        #         return self.choose_random_move(battle)
        # else:
        #     self.switchesInARow = 0
        #     #print(battle.available_moves)
        #     moves = [(move, self.embeddingModel.wv[move.id] if move.id in self.embeddingModel.wv.key_to_index else np.zeros(100)) for move in battle.available_moves]
        #     similarities = [(move, 1 - cosine(moveVector, query_vector)) for move, moveVector in moves]
        #     if(len(similarities) > 0):
        #         closest_move = max(similarities, key=lambda x: x[1])[0]

        #         # Calculate similarity to final performed move for weighting rewards later
        #         moveVectorPlusTeraSwitch = torch.tensor(self.embeddingModel.wv[closest_move.id] if closest_move.id in self.embeddingModel.wv.key_to_index else np.zeros(100))
        #         teraInt = 1 if tera and battle.can_tera else 0
        #         moveVectorPlusTeraSwitch = torch.cat((moveVectorPlusTeraSwitch, torch.tensor([teraInt, 0]))).to(device)
        #         self.similarities.append( F.cosine_similarity(logits, moveVectorPlusTeraSwitch, dim=0) )

        #         return self.create_order(order=closest_move, terastallize=tera and battle.can_tera)
        #     else:
        #         #print("no moves available, going random")
        #         return self.choose_random_move(battle)
        

    def describe_embedding(self) -> Space:
        #low = [-1 for i in range(5000)] + [-1 for i in range(500)] + [0 for i in range(6)] + [0 for i in range(42)] + [0 for i in range(6)] + [0 for i in range(6)] + [-1 for i in range(500)] + [0 for i in range(6)] + [0 for i in range(42)] + [0 for i in range(6)] + [0 for i in range(6)] + [-10 for i in range(70)]
        #high = [1 for i in range(5000)] + [1 for i in range(500)] + [1000 for i in range(6)] + [6 for i in range(42)] + [3 for i in range(6)] + [7 for i in range(6)] + [1 for i in range(500)] + [1000 for i in range(6)] + [6 for i in range(42)] + [3 for i in range(6)] + [7 for i in range(6)] + [0 for i in range(70)]
        low = [-1000 for i in range(6121)]
        high = [1000 for i in range(6121)]
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
            return np.array([0 for i in range(6151)]), {}

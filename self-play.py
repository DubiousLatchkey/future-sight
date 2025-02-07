import asyncio
from LSTMPlayer2 import LSTMPlayer, Agent
import torch
from gymnasium.utils.env_checker import check_env
from poke_env.player import RandomPlayer
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Self-play training script.')
parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations for training')
parser.add_argument('--load_model', type=str, help='Filename of the model to load')
args = parser.parse_args()

iterations = args.iterations


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = Agent()
if args.load_model:
    agent.load_model(args.load_model)

player = LSTMPlayer(agent)
player2 = LSTMPlayer(agent, playerString="p2a")

player3 = RandomPlayer()

async def battle():
    await player.battle_against(player3)



# train
agent.model.visualize()
for epoch in range(1, iterations + 1):

    asyncio.get_event_loop().run_until_complete(battle())

    # Train players
    player1Rewards, player1LogProbs, player1Values, player1SwitchingLosses, player1TeraLosses = player.popTrainingVariables()
    #player2Rewards, player2LogProbs, player2Values, player2SwitchingLosses, player2TeraLosses = player2.popTrainingVariables()

    agent.update_policy(player1Rewards, player1LogProbs, player1Values, player1SwitchingLosses, player1TeraLosses)
    #agent.update_policy(player2Rewards, player2LogProbs, player2Values, player2SwitchingLosses, player2TeraLosses)

    if epoch % 100 == 0:
        print(f"Game {epoch} finished.")
        if(not os.path.exists("checkpoints/")):
            os.makedirs("checkpoints/")
        agent.save_model(f"checkpoints/model_epoch_{epoch}.pth")

# Plotting the losses
epochs = range(1, iterations + 1)
losses = agent.pop_losses()  # Assuming agent has a method to get all losses

plt.figure(figsize=(20, 12))

plt.plot(epochs, losses['policy_losses'], label='Policy Loss')
plt.plot(epochs, losses['critic_losses'], label='Value Loss')
plt.plot(epochs, losses['switching_losses'], label='Switching Loss')
plt.plot(epochs, losses['tera_losses'], label='Tera Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Losses Over Time')
plt.legend()
plt.grid(True)
plt.savefig('training_losses.png')
plt.show()
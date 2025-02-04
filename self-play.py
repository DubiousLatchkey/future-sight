import asyncio
from LSTMPlayer2 import LSTMPlayer, Agent
import torch
from gymnasium.utils.env_checker import check_env
from poke_env.player import RandomPlayer
import argparse
import os

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
async def battle():
    await player.battle_against(player2)


# train
agent.visualize()
for epoch in range(1, iterations + 1):

    asyncio.get_event_loop().run_until_complete(battle())

    # Train players
    player1Rewards, player1Similarities = player.popRewardsAndSimilarities()
    player2Rewards, player2Similarities = player2.popRewardsAndSimilarities()

    agent.update_policy(player1Rewards, player1Similarities)
    agent.update_policy(player2Rewards, player2Similarities)

    if epoch % 100 == 0:
        print(f"Game {epoch} finished.")
        if(not os.path.exists("checkpoints/")):
            os.makedirs("checkpoints/")
        agent.save_model(f"checkpoints/model_epoch_{epoch}.pth")
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
agent.load_model("checkpoints/model_epoch_30000.pth")

agent2 = Agent()
agent2.load_model("checkpoints/model_epoch_100.pth")

player = LSTMPlayer(agent)
player2 = LSTMPlayer(agent2)
async def battle():
    battles = 100

    player3 = RandomPlayer(battle_format="gen9randombattle")
    await player.battle_against(player2, n_battles=battles)

    print(
        "Trained player won %d / %d battles against less trained"
        % (
            player.n_won_battles, battles
        )
    )


asyncio.get_event_loop().run_until_complete(battle())

from poke_env import AccountConfiguration
from poke_env.player import Player, RandomPlayer
from minimaxPlayer import MinimaxPlayer
import asyncio
import time
from llmPlayer import LLMPlayer
from NNPlayer import NNPlayer

# No authentication required
#my_account_config = AccountConfiguration("guy_1", None)
#player = Player(account_configuration=my_account_config)

# Auto-generated configuration for local use
#player = Player()

async def main():
    start = time.time()

    # We create two players.
    random_player = MinimaxPlayer(
        battle_format="gen8randombattle", use_random=False
    )
    nnPlayer = LLMPlayer(
        battle_format="gen8randombattle"
    )

    # Now, let's evaluate our player
    battles=1
    await nnPlayer.battle_against(random_player, n_battles=battles)

    print(
        "LLM player won %d / %d battles [this took %f seconds]"
        % (
            nnPlayer.n_won_battles, battles, time.time() - start
        )
    )


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
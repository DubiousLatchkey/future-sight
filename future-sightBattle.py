from poke_env import AccountConfiguration
from poke_env.player import Player, RandomPlayer
from minimaxPlayer import MinimaxPlayer
import asyncio
import time

# No authentication required
#my_account_config = AccountConfiguration("guy_1", None)
#player = Player(account_configuration=my_account_config)

# Auto-generated configuration for local use
#player = Player()

async def main():
    start = time.time()

    # We create two players.
    random_player = MinimaxPlayer(
        battle_format="gen9randombattle", use_random=True
    )
    minimax_player = MinimaxPlayer(
        battle_format="gen9randombattle", use_random=False
    )

    # Now, let's evaluate our player
    await minimax_player.battle_against(random_player, n_battles=2)

    print(
        "Minimax player won %d / 2 battles [this took %f seconds]"
        % (
            minimax_player.n_won_battles, time.time() - start
        )
    )


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
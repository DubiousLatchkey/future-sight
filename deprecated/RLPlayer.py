import numpy as np
from gym.spaces import Space, Box
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import Gen9EnvSinglePlayer
from poke_env.player import Player
from poke_env.teambuilder.teambuilder import Teambuilder
from poke_env.environment.battle import AbstractBattle
from poke_env.player.openai_api import ActType, ObsType, OpenAIGymEnv

class RLPlayer(Gen9EnvSinglePlayer):

    def __init__(self, opponent: Player | str | None, account_configuration: AccountConfiguration | None = None, *, avatar: int | None = None, battle_format: str | None = None, log_level: int | None = None, save_replays: bool | str = False, server_configuration: ServerConfiguration | None = None, start_listening: bool = True, start_timer_on_battle_start: bool = False, ping_interval: float | None = 20, ping_timeout: float | None = 20, team: str | Teambuilder | None = None, start_challenging: bool = True):
        super().__init__(opponent, account_configuration, avatar=avatar, battle_format=battle_format, log_level=log_level, save_replays=save_replays, server_configuration=server_configuration, start_listening=start_listening, start_timer_on_battle_start=start_timer_on_battle_start, ping_interval=ping_interval, ping_timeout=ping_timeout, team=team, start_challenging=start_challenging)

        # TODO: Get embeddings


    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )
    
    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)
    
    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

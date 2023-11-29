# -*- coding: utf-8 -*-
# Thanks Rempton Games

from poke_env.environment.move_category import MoveCategory
from poke_env.environment import AbstractBattle
from poke_env.environment import Pokemon
from poke_env.environment import Move
from poke_env.player.battle_order import BattleOrder
from poke_env.player.battle_order import DefaultBattleOrder

import json
import os

def calculate_damage(move, attacker, defender, pessimistic, is_bot_turn):
    if move is None:
        print("Why is move none?")
        return 0
    if move.category == MoveCategory.STATUS:
        return 0
    # Start with the base power of the move
    damage = move.base_power
    ratio = 1
    # Multiply by the attack / defense ratio
    if move.category == MoveCategory.PHYSICAL:
        # estimate physical ratio
        ratio = calculate_physical_ratio(attacker, defender, is_bot_turn)
    elif move.category == MoveCategory.SPECIAL:
        # estimate special ratio
        ratio = calculate_special_ratio(attacker, defender, is_bot_turn)
    damage = damage * ratio
    level_multiplier = ((2 * attacker.level) / 5 ) + 2
    damage = damage * level_multiplier
    # Finish calculating the base damage of the attack
    damage = (damage / 50) + 2;
    # Damage is multiplied by a random value between 0.85 and 1. Pessimistic flag gets a lower bound
    if pessimistic:
        damage = damage * 0.85
    if move.type == attacker.type_1 or move.type == attacker.type_2:
        damage = damage * 1.5
    type_multiplier = defender.damage_multiplier(move)
    damage = damage * type_multiplier
    # print(f"Damage calculation for move {move} against opponent {battle.opponent_active_pokemon} is {damage}")
    return damage

# The following two functions work very similarly, just focusing on different stats.
# They get the ratio between my Pokemon's attack and my opponent's estimated defense
# In random battles each Pokemon has 85 EVs in each stat and a neutral nature
# As far as I can tell IVs are random - assume average IVs (15)
def calculate_physical_ratio(attacker, defender, is_bot_turn):
    if is_bot_turn:
        # Get my attack value
        attack = attacker.stats["atk"]
        defense = 2 * defender.base_stats["def"]
        # I am adding 36 because it represents a very average amount of evs and ivs in the stat
        defense = defense + 36
        defense = ((defense * defender.level) / 100 ) + 5
    else:
        defense = defender.stats["def"]
        attack = 2 * attacker.base_stats["atk"]
        attack = attack + 36
        attack = ((attack * attacker.level) / 100) + 5
    return attack / defense   

def calculate_special_ratio(attacker, defender, is_bot_turn):
    if is_bot_turn:
        # Get my special attack value
        spatk = attacker.stats["spa"]
        spdef = 2 * defender.base_stats["spd"]
        # I am adding 36 because it represents a very average amount of evs and ivs in the stat
        spdef = spdef + 36
        spdef = ((spdef * defender.level) / 100 ) + 5
    else: 
        spdef = defender.stats["spd"]
        spatk = 2 * attacker.base_stats["spa"]
        spatk = spatk + 36
        spatk = ((spatk * attacker.level) / 100) + 5
    return spatk / spdef

def opponent_can_outspeed(my_pokemon, opponent_pokemon):
    my_speed = my_pokemon.stats["spe"]
    # Assume the worst - max IVs for opponent speed
    opponent_max_speed = 2 * opponent_pokemon.base_stats["spe"]
    # Add 52 - thats 31 for IVs and 21 for EVs (which are distributed evenly)
    opponent_max_speed = opponent_max_speed + 52
    opponent_max_speed = ((opponent_max_speed * opponent_pokemon.level) / 100) + 5
    if opponent_max_speed > my_speed: 
        return True
    else: 
        return False

def calculate_total_HP(pokemon, is_dynamaxed): 
    HP = pokemon.base_stats["hp"] * 2
    # Add average EVs and IVs to stat
    HP = HP + 36
    HP = ((HP * pokemon.level) / 100)
    HP = HP + pokemon.level + 10
    if is_dynamaxed: 
        HP = HP * 2
    return HP

# Returns a value that determines how well my_pokemon matches up with
# opponent_pokemon defensively. If opponent_pokemon has multiple types,
# return the value associated with the worse matchup
def get_defensive_type_multiplier(my_pokemon, opponent_pokemon):
    multiplier = 1
    first_type = opponent_pokemon.type_1
    first_multiplier = my_pokemon.damage_multiplier(first_type)
    second_type = opponent_pokemon.type_2
    if second_type is None:
        return first_multiplier
    second_multiplier = my_pokemon.damage_multiplier(second_type)
    multiplier = first_multiplier if first_multiplier > second_multiplier else second_multiplier
    return  multiplier

def get_pokemon_data(pokemon: (str, Pokemon)):
    pokemon_identifier = pokemon[0]

    pokemon = pokemon[1]
    pokemon_data = {}
    pokemon_data["identifier"] = pokemon_identifier
    pokemon_data["heightm"] = pokemon._heightm
    pokemon_data["possible_abilities"] = pokemon._possible_abilities
    pokemon_data["species"] = pokemon._species
    pokemon_data["type_1"] = pokemon._type_1.name 
    pokemon_data["type_2"] = pokemon._type_2.name if pokemon._type_2 else None
    pokemon_data["weightkg"] = pokemon._weightkg

    pokemon_data["ability"] = pokemon._ability
    pokemon_data["active"] = pokemon._active
    pokemon_data["gender"] = pokemon._gender.name
    pokemon_data["level"] = pokemon._level
    pokemon_data["max_hp"] = pokemon._max_hp
    pokemon_data["moves"] = [movesTuple[1].id for movesTuple in pokemon._moves.items()]

    pokemon_data["boosts"] = list(pokemon._boosts.items())
    pokemon_data["current_hp"] = pokemon._current_hp
    pokemon_data["effects"] = [(effectsTuple[0].name, effectsTuple[1]) for effectsTuple in pokemon._effects.items()]
    pokemon_data["first_turn"] = pokemon._first_turn
    pokemon_data["terastallized"] = pokemon._terastallized
    pokemon_data["terastallized_type"] = pokemon._terastallized_type.name if pokemon._terastallized_type else None
    pokemon_data["item"] = pokemon._item
    # pokemon_data["last_request"]
    # pokemon_data["last_details"]
    pokemon_data["must_recharge"] = pokemon._must_recharge
    pokemon_data["preparing_move"] = pokemon._preparing_move.id if pokemon._preparing_move else None
    # pokemon_data["preparing_target"] = pokemon._preparing_target
    pokemon_data["protect_counter"] = pokemon._protect_counter
    pokemon_data["revealed"] = pokemon._revealed
    pokemon_data["status"] = pokemon._status.name if pokemon._status else None
    pokemon_data["status_counter"] = pokemon._status_counter

    return pokemon_data


def dump_battle_data(battle: AbstractBattle, action: BattleOrder):
    battle_data = {}

    battle_data["dynamax_turn"] = battle._dynamax_turn
    battle_data["finished"] = battle._finished
    battle_data["turn"] = battle._turn
    battle_data["opponent_can_terrastallize"] = battle._opponent_can_terrastallize
    battle_data["opponent_dynamax_turn"] = battle._opponent_dynamax_turn
    battle_data["won"] = battle._won

    battle_data["weather"] = [(weatherTuple[0].name, weatherTuple[1]) for weatherTuple in battle._weather.items()]
    battle_data["fields"] = [(fieldsTuple[0].name, fieldsTuple[1]) for fieldsTuple in battle._fields.items()]
    battle_data["opponent_side_conditions"] = [(conditionTuple[0].name, conditionTuple[1]) for conditionTuple in battle._opponent_side_conditions.items()]
    battle_data["side_conditions"] = [(conditionTuple[0].name, conditionTuple[1]) for conditionTuple in battle._side_conditions.items()]
    battle_data["reviving"] = battle.reviving

    battle_data["team"] = [get_pokemon_data(pokemon) for pokemon in battle._team.items()]
    battle_data["opponent_team"] = [get_pokemon_data(pokemon) for pokemon in battle._opponent_team.items()]

    if(type(action) is DefaultBattleOrder):
        battle_data["action"] = None
        battle_data["action_type"] = None
    elif(type(action.order) is Move):
        battle_data["action"] = action.order.id
        battle_data["action_type"] = "move"
    elif(type(action.order) is Pokemon):
        battle_data["action_type"] = "switch"
        battle_data["action"] = action.order._species # Somewhat off from identifier, got to figure out how to get it


    data = json.dumps(battle_data, indent=4)

    if(not os.path.exists("gameStates/" + battle._battle_tag + "/") ):
        os.mkdir("gameStates/" + battle._battle_tag + "/")

    with open("gameStates/" + battle._battle_tag + "/" + battle.player_username + "_" + str(battle._turn) + ".json", "w+") as f:
        f.write(data)
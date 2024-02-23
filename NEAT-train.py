import neat
from poke_env.environment.battle import AbstractBattle
from NNPlayer import NNPlayer
import torch
import asyncio
import random
from minimaxPlayer import MinimaxPlayer
import graphviz
import warnings
import datetime
import os

# NN visualizer from NEATPython
def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot


# alternate version of NNPlayer that takes in NEAT model
class NEATPlayer(NNPlayer):
    def __init__(self, battle_format, NEATModel):
        super().__init__(battle_format)
        self.NEATModel = NEATModel

    def runInference(self, battle: AbstractBattle):
        # Embed your team
        myPokemon = sorted([i[1] for i in battle.team.items()], key= lambda x: x.species)
        pokemon1 = self.processPokemon(myPokemon[0]) if len(myPokemon) > 0 else self.processPokemon(None)
        pokemon2 = self.processPokemon(myPokemon[1]) if len(myPokemon) > 1 else self.processPokemon(None)
        pokemon3 = self.processPokemon(myPokemon[2]) if len(myPokemon) > 2 else self.processPokemon(None)
        pokemon4 = self.processPokemon(myPokemon[3]) if len(myPokemon) > 3 else self.processPokemon(None)
        pokemon5 = self.processPokemon(myPokemon[4]) if len(myPokemon) > 4 else self.processPokemon(None)
        pokemon6 = self.processPokemon(myPokemon[5]) if len(myPokemon) > 5 else self.processPokemon(None)

        # Embed opponent's team
        theirPokemon = sorted([i[1] for i in battle.opponent_team.items()], key= lambda x: x.species)
        opponentPokemon1 = self.processPokemon(theirPokemon[0]) if len(theirPokemon) > 0 else self.processPokemon(None)
        opponentPokemon2 = self.processPokemon(theirPokemon[1]) if len(theirPokemon) > 1 else self.processPokemon(None)
        opponentPokemon3 = self.processPokemon(theirPokemon[2]) if len(theirPokemon) > 2 else self.processPokemon(None)
        opponentPokemon4 = self.processPokemon(theirPokemon[3]) if len(theirPokemon) > 3 else self.processPokemon(None)
        opponentPokemon5 = self.processPokemon(theirPokemon[4]) if len(theirPokemon) > 4 else self.processPokemon(None)
        opponentPokemon6 = self.processPokemon(theirPokemon[5]) if len(theirPokemon) > 5 else self.processPokemon(None)

        #print(pokemon1.shape, opponentPokemon1.shape, opponentPokemon2.shape)
        # Concat with Conditions
        x = torch.concatenate((pokemon1, pokemon2, pokemon3, pokemon4, pokemon5, pokemon6, opponentPokemon1, opponentPokemon2, opponentPokemon3, opponentPokemon4, opponentPokemon5, opponentPokemon6))

        #print(x.shape)

        # This is hacked together, so forget making non torch version of the previous encoding
        x = self.NEATModel.activate(x.detach().numpy())

        return x

async def battle(player1, player2, n_battles=3):
    await player1.battle_against(player2, n_battles=n_battles)

async def battleWithVictor(player1, player2):
    player1Wins = player1.n_won_battles
    player2Wins = player2.n_won_battles
    await player1.battle_against(player2, n_battles=3)
    player1Wins = player1.n_won_battles - player1Wins
    player2Wins = player2.n_won_battles - player2Wins
    if(player1Wins >= 2):
        return 0
    if(player2Wins >= 2):
        return 1
    if(player1Wins > player2Wins):
        return 0
    if(player2Wins > player1Wins):
        return 1
    return 2 # Draw 

async def battles(coroutines):
    await asyncio.gather(*coroutines)

# round robin code
def fixtures(teams):
    if len(teams) % 2:
        teams.append('Bye')  
    rotation = list(teams)       # copy the list

    fixtures = []
    for i in range(0, len(teams)-1):
        fixtures.append(rotation)
        rotation = [rotation[0]] + [rotation[-1]] + rotation[1:-1]

    return fixtures


# run tournament
def evaluate_genomes(genomes, config):
    
    global genomesList, networks

    players = []
    genomesList = []
    networks = []

    # populate lists with new players
    for genomeID, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        networks.append(net)

        players.append(NEATPlayer(battle_format="gen9randombattle", NEATModel=net))
        genomesList.append(genome)
        genome.fitness = 0

    #print(genomes)

    # run round robin tournament
    # playerIndexes = [i for i in range(len(players))]
    # matches = fixtures(playerIndexes)
    # for f in matches:
    #     coroutines = []
    #     matchups = []
    #     for player1, player2 in zip(*[iter(f)]*2):
    #         matchups.append((player1, player2))
    #         coroutines.append(battle(players[player1], players[player2]))

    #     print("matchups:", matchups)
    #     asyncio.get_event_loop().run_until_complete(battles(coroutines))
    
    
    # Run single elim tournament
    playerIndexes = [i for i in range(len(players))]
    # Make list for storing top 1/4th
    topPlayersCutoff = len(playerIndexes) // 4
    topPlayers = []
    while(len(playerIndexes) > 1):

        # Make top players cutoff
        if(len(topPlayers) == 0 and len(playerIndexes) <= topPlayersCutoff):
            topPlayers = playerIndexes

        # Pair up
        coroutines = []
        pairs = [(playerIndexes[i], playerIndexes[i + 1]) for i in range(0, len(playerIndexes), 2)]
        print(pairs)
        for player1, player2 in pairs:
            coroutines.append(battle(players[player1], players[player2]))

        # Get existing wins to establish baseline
        baseWins = [players[i].n_won_battles for i in playerIndexes]
        
        asyncio.get_event_loop().run_until_complete(battles(coroutines))
    
        # Get new wins to get who won for each pair
        newWins = [players[i].n_won_battles for i in playerIndexes]

        # Find winners and replace playerIndexes with only winners for next round
        wins = [newWins[i] - baseWins[i] for i in range(len(baseWins))]
        winningIndexes = []
        for i in range(0, len(wins), 2):
            if(wins[i] > wins[i + 1]):
                winningIndexes.append(i)
            elif(wins[i + 1] > wins[i]):
                winningIndexes.append(i + 1)
            else:
                winningIndexes.append(random.choice((i, i + 1)))

        newPlayerIndexes = []
        for i in winningIndexes:
            newPlayerIndexes.append(playerIndexes[i])
        playerIndexes = newPlayerIndexes

    # Save image of model that wins tournament
    if(playerIndexes[0]):
        if(not os.path.exists("modelImages/") ):
            os.mkdir("modelImages/")
        draw_net(config=config, genome=genomesList[playerIndexes[0]], filename= "modelImages/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".gv")

    # Top players get additional chances against heuristic players
    minimaxPlayers = []
    coroutines = []
    for topPlayerIndex in topPlayers:
        minimaxPlayers.append(MinimaxPlayer(battle_format="gen9randombattle", use_random=False))
        coroutines.append(battle(players[topPlayerIndex], minimaxPlayers[-1], 10))

    asyncio.get_event_loop().run_until_complete(battles(coroutines))

    # update fitness
    for index, player in enumerate(players):
        genomesList[index].fitness += player.n_won_battles # - player.n_lost_battles
    
def main():
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config.txt"
    )

    # make models
    #population = neat.Population(config)
    population = neat.Checkpointer.restore_checkpoint("neat-checkpoint-7")

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    population.add_reporter(neat.Checkpointer(1, 900))

    # run training
    best = population.run(evaluate_genomes, 20)


    # evaluate
    #print('\nBest genome:\n{!s}'.format(best))

if __name__ == "__main__":
    main()
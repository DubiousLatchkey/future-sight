import torch
import numpy as np
import torch.nn as nn
import json
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance
from torch.utils.data import DataLoader, Dataset
import pickle
from sklearn.manifold import TSNE

# for filename in os.listdir("randomTeams/"):
#     with open("randomTeams/" + filename) as f:
#         text = f.read()
#         if("maushold" in text):
#             print(text)

pokemonToId = {}
idToPokemon = {}

class Pokemon_SkipGram_Model(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(Pokemon_SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=128,
            max_norm=1,
        )
        self.linear = nn.Linear(
            in_features=128,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x
   
class PokemonTeamsDataSet(Dataset):
    """Dataset of teams"""

    def __init__(self, transform=None):
        self.transform = transform

        teams = []
        self.data = []
        self.uniquePokemon = set()
        for filename in os.listdir("randomTeams/"):
            with open("randomTeams/" + filename) as f:
                team = f.read().split(" ")
                self.uniquePokemon = self.uniquePokemon.union(set(team))
                teams.append(team)

        count = 1
        for pokemon in sorted(list(self.uniquePokemon)):
            pokemonToId[pokemon] = count
            count += 1
        pokemonToId["<unk>"] = 0

        
        for i in pokemonToId.items():
            idToPokemon[i[1]] = i[0]

        #print(idToPokemon, pokemonToId)
        print("number of teams:",len(teams))
        for team in teams:
            self.data += makeTeamSkipGrams(team)
        print("number of skipgrams:",len(self.data))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
        

def processPokemonName(pokemon):
    return pokemon.lower().replace(" ", "").replace("-", "").replace(",", "")


def makeTeamSkipGrams(team):
    skipgrams = []
    for index in range(len(team)):
        pokemon = team[index]
        for otherPokemon in team[:index] + team[index + 1:]:
            #print(pokemon, otherPokemon)
            skipgrams.append((torch.tensor(pokemonToId[pokemon], dtype=torch.long), torch.tensor(pokemonToId[otherPokemon], dtype=torch.long)))

    return skipgrams


def train(training_dataloader, model, loss_function, optimizer, epochs):
    training_losses = []
    training_accuracies = []
    for i in range(epochs):
        size = len(training_dataloader.dataset)
        batch_training_loss = 0
        correct = 0
        for batch, (x, y) in enumerate(training_dataloader):        
            x = x.cuda()
            y = y.cuda()
            #print(x)
            #print(y)
            pred = model(x)
            loss = loss_function(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_training_loss += loss
            
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
        training_losses.append(batch_training_loss.item() / size)
        training_accuracies.append(correct / size)
            
       
        print(f"Epoch: {i+1} Avg loss: {training_losses[-1]:>8f} \n")

    
    return model, training_losses, training_accuracies

def get_distance_matrix(wordvecs, metric):
    dist_matrix = distance.squareform(distance.pdist(wordvecs, metric))
    return dist_matrix

def get_k_similar_words(word, dist_matrix, k=10):
    idx = pokemonToId[word]
    dists = dist_matrix[idx]
    ind = np.argpartition(dists, k)[:k+1]
    ind = ind[np.argsort(dists[ind])][1:]
    out = [(i, idToPokemon[i], dists[i]) for i in ind]
    return out
    

if __name__ == "__main__":
    torch.manual_seed(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))



    dataset = PokemonTeamsDataSet()
    print("Number of pokemon forms:", len(pokemonToId.keys()))
    # Train
    model = Pokemon_SkipGram_Model(vocab_size=len(pokemonToId.keys())).to(device)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()

    model, t_loss, t_a,= train(dataloader, model, loss_function, optimizer, 10)

    plt.plot(range(10), t_loss, label="training")
    plt.savefig("pokemonLoss.png")
    torch.save(model.state_dict(), "pokemonModel.pt")
    plt.clf()

    # Test
    model.load_state_dict(torch.load("pokemonModel.pt"))
    wordvecs = model.linear.weight.cpu().detach().numpy()
    dmat = get_distance_matrix(wordvecs, 'cosine')

    tokens = ['maushold', 'blissey', 'arceusgrass', 'arceuswater', 'mausholdfour']
    for word in tokens:
        print(word, [t[1] for t in get_k_similar_words(word, dmat)], "\n")

    tsne = TSNE()
    embed_tsne = tsne.fit_transform(wordvecs[:-1, :])

    fig, ax = plt.subplots(figsize=(32, 32))
    for idx in range(len(embed_tsne)):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        plt.annotate(idToPokemon[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)

    plt.savefig("pokemonEmbedding.png")

    with open("pokemonToId.pickle", "wb") as f:
        pickle.dump(pokemonToId, f, protocol=pickle.HIGHEST_PROTOCOL)
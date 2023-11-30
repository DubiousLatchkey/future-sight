import torch
import numpy as np
import torch.nn as nn
import json
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance


# Thanks Olga Chernytska

SKIPGRAM_N_WORDS = 4
MAX_SEQUENCE_LENGTH = 256
from torch.utils.data import DataLoader, Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class SkipGram_Model(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()
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
    
class MoveSentenceDataset(Dataset):
    """Dataset of move sentences"""

    def __init__(self, transform=None):
        self.transform = transform

        self.data = []
        for filename in os.listdir("onlineReplayMoveSentences/"):
            with open("onlineReplayMoveSentences/" + filename) as f:
                self.data += windowizer(f.read().split(" "))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
    
def indexToOneHotTensor(moveIndex, totalMoves):
    tensor = torch.zeros(totalMoves)  
    tensor[moveIndex] = 1
    return tensor


moveToIndex = {}

# Thanks Musashi Hinck
def windowizer(row, wsize=4):
    """
    Windowizer function for Word2Vec. Converts sentence to sliding-window
    pairs.
    """
    if(len(row) <= 1):
        return []
    doc = row
    out = []
    for i, wd in enumerate(doc):
        target = moveToIndex[wd]
        window = [i+j for j in
                  range(-wsize, wsize+1, 1)
                  if (i+j>=0) &
                     (i+j<len(doc)) &
                     (j!=0)]

        out+=[(torch.tensor(target, dtype=torch.long), torch.tensor(moveToIndex[doc[w]], dtype=torch.long)) for w in window]
    row = out
    return row

def train(training_dataloader, model, loss_function, optimizer, epochs):
    training_losses = []
    training_accuracies = []
    for i in range(epochs):
        size = len(training_dataloader.dataset)
        batch_training_loss = 0
        correct = 0
        for batch, (x, y) in enumerate(training_dataloader):        
            #x = x.cuda()
            #y = y.cuda()
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
    idx = moveToIndex[word]
    dists = dist_matrix[idx]
    ind = np.argpartition(dists, k)[:k+1]
    ind = ind[np.argsort(dists[ind])][1:]
    out = [(i, indexToMove[i], dists[i]) for i in ind]
    return out
    
torch.manual_seed(1)

with open("venv\Lib\site-packages\poke_env\data\static\moves\gen9moves.json") as f:
    data = json.load(f)
    count = 0
    for i in data:
        moveToIndex[i] = count
        count += 1

indexToMove = {}
for i in moveToIndex.items():
    indexToMove[i[1]] = i[0]

#dataset = MoveSentenceDataset()

#print(dataset.__getitem__(5), dataset.__getitem__(6))
#print(indexToMove[dataset.__getitem__(5)[0]], indexToMove[dataset.__getitem__(5)[1]])

model = SkipGram_Model(len(moveToIndex.keys())).to(device)

#dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()

# training
#model, t_loss, t_a,= train(dataloader, model, loss_function, optimizer, 20)

#plt.plot(range(20), t_loss, label="training")
#plt.savefig("moveLoss.png")
#torch.save(model.state_dict(), "moveModel.pt")


# testing
model.load_state_dict(torch.load("moveModel.pt"))
wordvecs = model.linear.weight.cpu().detach().numpy()
dmat = get_distance_matrix(wordvecs, 'cosine')

tokens = ['toxic', 'thunderbolt', 'thunder', 'protect', 'sunnyday', 'counter']
for word in tokens:
    print(word, [t[1] for t in get_k_similar_words(word, dmat)], "\n")
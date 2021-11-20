"""Feed-forward neural network."""


import pandas as pd
import os
import sys
import json
import torch
from torch import nn
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
import warnings
from utils import model


warnings.filterwarnings("ignore")


# DEFINE GLOBAL VARIABLES
currentDir = sys.path[0]
dataPath = "data/"
modelPath = "model/"
processedPath = f"{dataPath}/processed/"
plotsPath = "plot/"
edaPlotsPath = f"{plotsPath}/eda/"

trainFileName = "trainTab.csv"
testFileName = "test.csv"
catClassFileName = "cat_class.csv"
scoresFileName = "aesth_scores.json"

trainFullPath = os.path.join(processedPath, trainFileName)
testFullPath = os.path.join(processedPath, testFileName)
classFullPath = os.path.join(processedPath, catClassFileName)
scoresFullPath = os.path.join(processedPath, scoresFileName)

# LOAD TABULAR DATA

train = pd.read_csv(trainFullPath)
test = pd.read_csv(testFullPath)
cat = pd.read_csv(classFullPath)

with open(scoresFullPath, 'r') as dataFile:
    data = json.load(dataFile)

aesth = pd.DataFrame(data)
train['Is Cat'] = cat['is_cat']
train['Aesth Score'] = aesth['mean_score_prediction']


class PawpularityDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, scale_data=True):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            if scale_data:
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# Prepare Boston dataset
X, y = model.regression_train_preprocess(train)

dataset = PawpularityDataset(X, y)
BATCH_SIZE = 64

loader = Data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, num_workers=2,
    )


class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            torch.nn.Linear(1, 200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(200, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, 1),
        )

    def forward(self, x):
        return self.layers(x)


# Initialize the MLP
mlp = MultiLayerPerceptron()
EPOCH = 200
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
loss_function = torch.nn.MSELoss()

# Run the training loop
for epoch in range(0, EPOCH):  # 5 epochs at maximum
    # Print epoch
    print(f'Starting epoch {epoch+1}')
    # Set current loss value
    current_loss = 0.0
    # Iterate over the DataLoader for training data
    for i, data in enumerate(loader, 0):
        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))
        # Zero the gradients
        optimizer.zero_grad()
        # Perform forward pass
        outputs = mlp(inputs)
        # Compute loss
        loss = loss_function(outputs, targets)
        # Perform backward pass
        loss.backward()
        # Perform optimization
        optimizer.step()
        # Print statistics
        current_loss += loss.item()
        if i % 10 == 0:
            print('Loss after mini-batch %5d: %.3f' %
                (i + 1, current_loss / 500))
            current_loss = 0.0
    # Process is complete.
    print('Training process has finished.')

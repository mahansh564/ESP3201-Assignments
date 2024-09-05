from ann import *
import pandas as pd

split = 0.8
dataset = pd.read_csv('data/ANN_circle.csv')

train = dataset.sample(frac=split)
test = dataset.drop(train.index)
print(test)

nn = NeuralNetwork(2, 4, 1)

for _ in range(1000):
    for inputs, target in zip(train[["x1", "x2"]].values, train["label"].values):
        nn.forward(inputs)
        nn.backward([target], inputs)

for inputs, target in zip(test[["x1", "x2"]].values, test["label"].values):
    print(nn.predict(inputs), target)

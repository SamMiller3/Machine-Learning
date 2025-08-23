# IMDB Reviews sentiment analysis using a feedforward network
# 23/08/25

# libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# model definition

class Model(nn.Module):
    def __init__(self, in_features=10000, h1=128, h2=64, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
# Load data
data = pd.read_csv("IMDB Dataset.csv")
data['sentiment'] = data['sentiment'].replace('positive', 1.0)
data['sentiment'] = data['sentiment'].replace('negative', 0.0)
texts = data['review'].values
labels = data['sentiment'].values
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

X_train = torch.FloatTensor(X_train.toarray())
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test.toarray())
y_test = torch.FloatTensor(y_test)

# Use DataLoader for mini-batch training
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

# Model, loss, optimizer
model = Model()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        predictions = (outputs > 0.5).long()
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

print(f"Test Accuracy: {100 * correct/total:.2f}%")

# allow user to interact/test

response = ""
print("type in your own movie reviews, type stop to end the program, it will classify whether it is positive or negative.")
while response.lower() != "stop":
    response = input("enter here: ")
    response_vector = vectorizer.transform([response])
    response_vector = torch.FloatTensor(response_vector.toarray())
    with torch.no_grad():
        output = model(response_vector)
    if output > 0.5:
        print("positive")
    else:
        print("negative")

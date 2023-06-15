import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = nn.functional.softmax(out, dim=1)
        return out


class MultiLayerBiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes) # *2 to account for bidirectional LSTM

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(x.device) # *2 to account for bidirectional LSTM
        c0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(x.device) # *2 to account for bidirectional LSTM
        # Forward propagate bidirectional LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = nn.functional.softmax(out, dim=1)
        return out


# Load the features and labels from numpy arrays
train_features = torch.from_numpy(np.load('train_features.npy')).float()
train_labels = torch.from_numpy(np.load('train_labels.npy'))#.long()
idx = np.random.permutation(len(train_features))
train_features, train_labels = train_features[idx], train_labels[idx]

test_features = torch.from_numpy(np.load('test_features.npy')).float()
test_labels = torch.from_numpy(np.load('test_labels.npy'))#.long()

# Define the LSTM parameters
input_size = train_features.shape[-1]
hidden_size = 128
num_classes = len(np.unique(train_labels))
num_frames = 15
# Instantiate the LSTM model
model = LSTMClassifier(input_size, hidden_size, num_classes).cuda()
model = MultiLayerBiLSTMClassifier(input_size, hidden_size, 2, num_classes).cuda()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the LSTM model
num_epochs = 100
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_total = 0
    train_correct = 0
    for i in range(0, len(train_features), num_frames):
        # Get a batch of 15 frames features and labels
        batch_features = train_features[i].cuda()
        batch_labels = train_labels[i].cuda()
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward + backward + optimize
        outputs = model(batch_features.unsqueeze(0))
        loss = criterion(outputs, batch_labels.unsqueeze(0))
        loss.backward()
        optimizer.step()
        # Accumulate training loss and accuracy
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += 1
        train_correct += (predicted == batch_labels).sum().item()
    train_loss /= train_total
    train_accuracy = 100 * train_correct / train_total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Testing
    model.eval()
    test_loss = 0.0
    test_total = 0
    test_correct = 0
    with torch.no_grad():
        for i in range(0, len(test_features), num_frames):
            # Get a batch of 15 frames features and labels
            batch_features = test_features[i].cuda()
            batch_labels = test_labels[i].cuda()
            # Forward
            outputs = model(batch_features.unsqueeze(0))
            loss = criterion(outputs, batch_labels.unsqueeze(0))
            # Accumulate testing loss and accuracy
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_total += 1
            if predicted == batch_labels:
                test_correct += 1
            # Calculate testing accuracy
        test_accuracy = test_correct / test_total
        test_loss /= test_total
        test_accuracy = 100* test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        # Print testing loss and accuracy
        print('Train Loss: {:.4f}, Train Accuracy: {:.2f}%  Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format( train_loss, train_accuracy, test_loss, test_accuracy))

# Plot training and testing losses and accuracies
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

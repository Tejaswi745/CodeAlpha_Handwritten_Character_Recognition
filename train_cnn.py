import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.rot90(x, 1, [1,2])),
    transforms.Lambda(lambda x: torch.flip(x, [2]))
])

# Dataset
train_dataset = datasets.EMNIST(
    root='./data',
    split='letters',
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# CNN MODEL
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(64*5*5, 128)
        self.fc2 = nn.Linear(128, 26)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(-1, 64*5*5)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

model = CNNModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TRAIN
epochs = 20

for epoch in range(epochs):
    for images, labels in train_loader:
        labels = labels - 1

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# SAVE
torch.save(model.state_dict(), "cnn_model.pth")

print("✅ CNN Model Saved")
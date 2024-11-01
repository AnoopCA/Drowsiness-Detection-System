import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Define the model in PyTorch
class DrowsinessDetectionModel(nn.Module):
    def __init__(self):
        super(DrowsinessDetectionModel, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = torch.relu(self.conv2(x))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.relu(self.conv4(x))
        x = self.pool(torch.relu(self.conv5(x)))
        x = self.pool(torch.relu(self.conv6(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# Data transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
train_dataset = datasets.ImageFolder(r"D:\ML_Projects\Drowsiness-Detection-System\Data\Driver Drowsiness Dataset\train", transform=transform)
test_dataset = datasets.ImageFolder(r"D:\ML_Projects\Drowsiness-Detection-System\Data\Driver Drowsiness Dataset\test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
model = DrowsinessDetectionModel()
criterion = nn.BCELoss()  # For binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().to(device)
        labels = labels.view(-1, 1)  # Reshape for binary classification output

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Save the model
torch.save(model.state_dict(), r"D:\ML_Projects\Drowsiness-Detection-System\Models\drowse_model_pt_1.pth")

print("Model training and saving completed.")

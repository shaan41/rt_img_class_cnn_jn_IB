import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        dim1 = 100
        dim2 = 50
        dim3 = 25

        super(SimpleCNN, self).__init__()
        # First convolutional layer (input channels, output channels, kernel size)
        self.conv1 = nn.Conv2d(3, dim1, kernel_size=3, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(dim1, dim2, kernel_size=3, padding=1)
        # Third convolutional layer
        self.conv3 = nn.Conv2d(dim2, dim3, kernel_size=3, padding=1)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        flattened_size = (
            28 * 28 * dim3
        )  # Update this based on your actual input and architecture
        self.fc1 = nn.Linear(
            flattened_size, 512
        )  # Example: 512 can be changed based on design
        self.fc2 = nn.Linear(512, num_classes)  # Output layer: number of classes

    def forward(self, x):
        # Apply conv1 with a ReLU activation, followed by max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply conv2 with a ReLU activation, followed by max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Apply conv3 with a ReLU activation, followed by max pooling
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the output for the dense layers
        x = x.view(x.shape[0], -1)
        # Apply fc1 with a ReLU activation
        x = F.relu(self.fc1(x))
        # Apply fc2
        x = self.fc2(x)
        return x

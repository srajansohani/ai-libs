import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------
# Create folders
# -----------------------------

os.makedirs("kernels", exist_ok=True)
os.makedirs("activation_patterns", exist_ok=True)

# -----------------------------
# Dataset
# -----------------------------

transform = transforms.ToTensor()

dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

sample_img, sample_label = dataset[0]
sample_img = sample_img.unsqueeze(0)

print("Sample label:", sample_label)

# -----------------------------
# CNN Model
# -----------------------------

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)

        self.relu = nn.ReLU()

        self.fc = nn.Linear(32 * 16 * 16, 10)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


model = SimpleCNN()

# -----------------------------
# Kernel Visualization (Grayscale)
# -----------------------------

def visualize_kernels(layer, epoch, layer_name):

    weights = layer.weight.data.clone().cpu()

    num_kernels = weights.shape[0]

    fig, axes = plt.subplots(1, num_kernels, figsize=(num_kernels*2,2))

    if num_kernels == 1:
        axes = [axes]

    for i in range(num_kernels):

        kernel = weights[i]

        if kernel.shape[0] > 1:
            kernel = kernel.mean(dim=0)
        else:
            kernel = kernel[0]

        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())

        axes[i].imshow(kernel, cmap="gray")

        axes[i].set_title(f"K{i}")
        axes[i].axis("off")

    plt.suptitle(f"{layer_name} kernels epoch {epoch}")

    path = f"kernels/{layer_name}_epoch_{epoch}.png"

    plt.savefig(path)
    plt.close()

    print("Saved:", path)

# -----------------------------
# Feature Map Hook
# -----------------------------

activations = {}

def get_activation(name):

    def hook(model, input, output):
        activations[name] = output.detach()

    return hook


model.conv1.register_forward_hook(get_activation("conv1"))
model.conv2.register_forward_hook(get_activation("conv2"))
model.conv3.register_forward_hook(get_activation("conv3"))

# -----------------------------
# Feature Map Visualization
# -----------------------------

def visualize_feature_maps(layer_name):

    fmap = activations[layer_name].squeeze()

    num_maps = fmap.shape[0]

    fig = plt.figure(figsize=(12,6))

    for i in range(num_maps):

        ax = fig.add_subplot(2, num_maps//2 + 1, i+1)

        ax.imshow(fmap[i], cmap="gray")

        ax.set_title(f"{layer_name}-{i}")
        ax.axis("off")

    plt.show()

# -----------------------------
# Activation Heatmap Overlay
# -----------------------------

def show_activation_overlay(layer_name, filter_index):

    fmap = activations[layer_name][0, filter_index]

    fmap = fmap.numpy()

    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())

    fmap = np.uint8(255 * fmap)

    fmap = np.resize(fmap, (28,28))

    plt.figure(figsize=(6,3))

    plt.subplot(1,2,1)
    plt.title("Original Digit")
    plt.imshow(sample_img.squeeze(), cmap="gray")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Activation Heatmap")

    plt.imshow(sample_img.squeeze(), cmap="gray")
    plt.imshow(fmap, cmap="jet", alpha=0.5)

    plt.axis("off")

    plt.show()

# -----------------------------
# Activation Maximization
# -----------------------------

def visualize_filter(model, layer, filter_index, name):

    img = torch.randn(1,1,28,28, requires_grad=True)

    optimizer = torch.optim.Adam([img], lr=0.1)

    for step in range(100):

        optimizer.zero_grad()

        x = img

        for module in model.children():

            x = module(x)

            if module == layer:
                break

        activation = x[0, filter_index]

        loss = -activation.mean()

        loss.backward()

        optimizer.step()

    result = img.detach().squeeze()

    plt.imshow(result, cmap="gray")
    plt.axis("off")

    path = f"activation_patterns/{name}_filter_{filter_index}.png"

    plt.savefig(path)
    plt.close()

    print("Saved:", path)

# -----------------------------
# Training Setup
# -----------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# BEFORE TRAINING
# -----------------------------

print("\nVisualizing kernels BEFORE training")

visualize_kernels(model.conv1, 0, "conv1")

# -----------------------------
# TRAINING
# -----------------------------

epochs = 5

for epoch in range(epochs):

    total_loss = 0

    for images, labels in loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss:", total_loss)

    visualize_kernels(model.conv1, epoch+1, "conv1")

# -----------------------------
# Forward pass for visualization
# -----------------------------

model.eval()

with torch.no_grad():

    output = model(sample_img)

pred = output.argmax()

print("Prediction:", pred.item())

# -----------------------------
# Feature Maps
# -----------------------------

visualize_feature_maps("conv1")
visualize_feature_maps("conv2")
visualize_feature_maps("conv3")

# -----------------------------
# Activation Heatmaps
# -----------------------------

show_activation_overlay("conv1", 0)
show_activation_overlay("conv1", 3)
show_activation_overlay("conv2", 5)

# -----------------------------
# Activation Maximization
# -----------------------------

print("\nGenerating filter patterns")

for i in range(model.conv1.out_channels):

    visualize_filter(model, model.conv1, i, "conv1")
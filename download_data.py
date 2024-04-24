import os
import torch
from torchvision import datasets
import pandas as pd

# Create directory if it doesn't exist
os.makedirs("MNIST_CSV", exist_ok=True)

# Download MNIST dataset
train_dataset = datasets.MNIST(root="./", train=True, download=True)
test_dataset = datasets.MNIST(root="./", train=False, download=True)

# Convert datasets to pandas DataFrames
train_df = pd.DataFrame(train_dataset.data.numpy(), columns=[f"pixel_{i}" for i in range(784)])
train_df["label"] = train_dataset.targets.numpy()

test_df = pd.DataFrame(test_dataset.data.numpy(), columns=[f"pixel_{i}" for i in range(784)])
test_df["label"] = test_dataset.targets.numpy()

# Save DataFrames to CSV files
train_df.to_csv("MNIST_CSV/train.csv", index=False)
test_df.to_csv("MNIST_CSV/test.csv", index=False)
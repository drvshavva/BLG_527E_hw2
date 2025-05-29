import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# 1. Dataset Preparation
def load_fashion_mnist():
    """Load and prepare Fashion-MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor()  # Keep in [0, 1] range for Sigmoid output
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader, test_loader

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trds = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform)
    teds = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(trds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(teds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, trds.classes

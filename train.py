import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import FashionCNN
from utils import get_data_loaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trload, _, _ = get_data_loaders(batch_size=64)
    model = FashionCNN().to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)
    epochs = 5
    for x in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in trload:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            outputs = model(imgs)
            loss    = crit(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(trload)
        print(f"Epoch {x+1}/{epochs} — Loss: {avg_loss:.4f}")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/fashion_cnn.pth")
    print("Model saved to models/fashion_cnn.pth")
if __name__ == "__main__":
    main()

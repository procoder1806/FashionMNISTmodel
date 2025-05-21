import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import FashionCNN
from utils import get_data_loaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, testl, classn = get_data_loaders(batch_size=64)

    model = FashionCNN().to(device)
    model.load_state_dict(torch.load("models/fashion_cnn.pth", map_location=device))
    model.eval()

    p, l = [], []
    with torch.no_grad():
        for imgs, labels in testl:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            p.extend(preds)
            l.extend(labels.numpy())

    # Confusion matrix
    cm = confusion_matrix(l, p)
    disp = ConfusionMatrixDisplay(cm, display_labels=classn)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("FashionMNIST Confusion Matrix")
    plt.show()

    # Overall accuracy
    acc = np.mean(np.array(p) == np.array(l))
    print(f"Test Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()

import torch
import matplotlib.pyplot as plt
from model import FashionCNN
from utils import get_data_loaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, tl, cln = get_data_loaders(batch_size=1)
    model = FashionCNN().to(device)
    model.load_state_dict(torch.load("models/fashion_cnn.pth", map_location=device))
    model.eval()
    i=int(input("Enter image number : "))
    testdata = tl.dataset
    img, label = testdata[i]
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"True: {cln[label]}")
    plt.axis('off')
    plt.show()
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
    pred_label = output.argmax(dim=1).item()
    print(f"Predicted: {cln[pred_label]}")

if __name__ == "__main__":
    main()

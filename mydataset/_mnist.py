from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms as trans

train_dataset = MNIST(root=r'D:\data', train=True, download=False,
                      transform=trans.Compose([trans.ToTensor(), trans.Resize(1500)]))
val_dataset = MNIST(root=r'D:\data', train=False, download=False,
                    transform=trans.Compose([trans.ToTensor(), trans.Resize(1500)]))
train_dl = DataLoader(dataset=train_dataset, batch_size=128)
val_dl = DataLoader(dataset=val_dataset, batch_size=128)

if __name__ == '__main__':
    for image, label in train_dl:
        print(image.shape)
        print(label)
        exit(0)

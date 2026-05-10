from operator import index
import numpy as np
import pandas as pd
import pathlib

from sympy.printing.latex import other_symbols
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Lambda # Working with PIL Images
"""
    ToTensor → Convert a PIL Image or ndarray to tensor and scale the values accordingly.
    Lambda → Apply a user-defined lambda as a transform.
"""


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        with open(data_path, mode='r') as f:
            self.df = pd.read_csv(f)

            self.features = self.df.drop(labels="label", axis=1)
            self.labels = self.df.loc[:, "label"]
        
        # Alwaws remember to transform them into tensors because dataloader only work with torch tensors
        self.transform = transform
        self.target_transform = target_transform

        # Applying transformation
        self.features = self.transform(self.features.to_numpy()) # torch.tensor(arr) does not work with dataframes
        self.labels = self.target_transform(torch.tensor(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.features[index], self.labels[index])


def scatter_mechanism(index: torch.Tensor, dim: torch.int = 1, value: torch.int = 1):
    """
        dim = 0 -> column wise
        dim = 1 -> row wise

        It will make more sence when
            If the self tensor is of shape [n, m] and dimention = 1 (shape of index becomes [m, 1])
            If the self tensor id of shape [n, m] and dimention = 0 (shape of index becomes [1, m])
        
        Its like pick the value from index and place it in that location of self tensor
    """


    one_hot_encoded = torch.zeros(size=[3, 3], dtype=torch.float) # num_rows = 3, num_classes = 3
    index = index.unsqueeze(dim = dim) # Must be same

    return one_hot_encoded.scatter(
        dim=dim,
        index=index,
        value=value,
    )


def main():
    home_dir = pathlib.Path(__file__).parent
    train_path = home_dir / "data" / "fashion-mnist_train.csv"
    test_path = home_dir / "data" / "fashion-mnist_test.csv"

    """ PyTorch Officials
        # Downloading the dataset → Not in csv form
        train_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,

            # You can use `Transform` here as well
            transform=ToTensor(),
            target_transform=Lambda(lambda y: torch.zeros(size=10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
        )

        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,

            # You can use `Transform` here as well
            transform=ToTensor(),
            target_transform=Lambda(lambda y: torch.zeros(size=10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
        )

        print(test_data)
    """

    transform = Lambda(lambda y: torch.tensor(y))
    target_transform = Lambda(lambda y: F.one_hot(y, 10))

    training_data = CustomDataset(data_path=train_path, transform=transform, target_transform=target_transform)
    testing_data = CustomDataset(data_path=test_path, transform=transform, target_transform=target_transform)

    train_dataloader = DataLoader(
        dataset=training_data,
        batch_size=15,
        shuffle=True
    )

    test_dataloader = DataLoader(
        dataset=testing_data,
        batch_size=5,
        shuffle=False
    )
    # These dataloader will return the batch and not the individual records

    # Iterating over training data
    for index, (features, label) in enumerate(train_dataloader):
        print("index : ", index)
        print("Features : ", features)
        print("Label : ", label)
        print("-" * 100)

        if index == 5: break


if __name__ == "__main__":
    main()

    """ 
        print(scatter_mechanism(
            index=torch.tensor([1, 2, 0]), # Must be 3 categories for visualizetion
            dim=1, # Change the value of dimention to see how it works 
            value=1
        ))
    """
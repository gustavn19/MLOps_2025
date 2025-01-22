import os

import matplotlib.pyplot as plt
import torch
import typer
from data import PokeData


def dataset_statistics(datadir: str = os.path.join(os.getcwd(), "data")) -> None:
    """Compute dataset statistics.

    Args:
    - datadir (str): The directory containing the dataset.
    """
    train_dataset = PokeData(datadir, 1)._get_train_loader()
    val_dataset = PokeData(datadir, 1)._get_val_loader()
    test_dataset = PokeData(datadir, 1)._get_test_loader()
    print(f"Train dataset")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print("\n")
    print(f"Validation dataset")
    print(f"Number of images: {len(val_dataset)}")
    print(f"Image shape: {val_dataset[0][0].shape}")
    print("\n")
    print(f"Test dataset")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")

    # Plot some examples
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(train_dataset[i][0].permute(1, 2, 0))
        plt.title(f"Label: {train_dataset[i][1]}")
        plt.axis("off")
    plt.savefig("pokemon_examples.png")
    plt.close()

    train_label_distribution = torch.bincount(train_dataset.target)
    test_label_distribution = torch.bincount(test_dataset.target)

    plt.bar(torch.arange(10), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("train_label_distribution.png")
    plt.close()

    plt.bar(torch.arange(10), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("test_label_distribution.png")
    plt.close()


if __name__ == "__main__":
    typer.run(dataset_statistics)

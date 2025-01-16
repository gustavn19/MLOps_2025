import os
import random
from pathlib import Path

import pandas as pd
import torch
import typer
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from tqdm import tqdm


def split_data_and_preprocess(raw_data_path: Path = Path("data/raw/dataset"),
               output_folder: Path = Path("data/processed"),
               split_ratio: tuple[float, float, float] = (1/3, 1/3, 1/3),
               image_size: tuple[int, int] = (128, 128)
               ) -> None:
    """
    Splits the dataset in `raw_data_path` into train, val, and test splits, 
    ensuring each class is equally represented, and stores them in `output_folder`.
    Before splitting the images are converted to .pt format.
    
    Args:
    - raw_data_path (str): Path to the source directory containing subfolders for classes.
    - output_folder (str): Path to the destination directory for the splits.
    - split_ratio (tuple): Ratios for train, val, and test splits. Default is 1/3 each.
    - image_size (tuple): Target image size for resizing. Default is 128x128 - current image size.
    """
    assert sum(split_ratio) == 1, "Split ratios must sum to 1."

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    datasets = {'train': [], 'val': [], 'test': []}
    labels = {'train': [], 'val': [], 'test': []}
    class_to_idx = {}

    # Assign a numerical label to each class
    class_names = sorted(os.listdir(raw_data_path))
    for idx, class_name in enumerate(class_names):
        class_to_idx[class_name] = idx

    # Process each class folder
    for class_name, class_idx in tqdm(class_to_idx.items()):
        class_path = os.path.join(raw_data_path, class_name)
        if not os.path.isdir(class_path):
            continue  # Skip if not a directory

        # Get all image files in the class folder
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(images)  # Shuffle for randomness

        # Compute split sizes
        n = len(images)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])

        # Split images
        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        # Process each split
        for split_name, split_images in splits.items():
            for image_name in split_images:
                # Transform image
                image_path = os.path.join(class_path, image_name)
                img = Image.open(image_path).convert('RGB')
                img_tensor = transform(img)
                    
                # Add to dataset
                datasets[split_name].append(img_tensor)
                labels[split_name].append(class_idx)
                
    # Save each split as a single .pt file
    for split_name in datasets.keys():
        images_tensor = torch.stack(datasets[split_name])  # Shape: [N, C, H, W]
        labels_tensor = torch.tensor(labels[split_name])   # Shape: [N]
        save_path = os.path.join(output_folder, f"{split_name}.pt")
        torch.save({'images': images_tensor, 'labels': labels_tensor}, save_path)


class PokeData(Dataset):
    """A PyTorch Dataset for the Pokemon dataset."""

    def __init__(self, data_path: Path, batch_size: int = 32) -> None:
        self.data_path = data_path
        self.train_path = os.path.join(data_path, "processed")
        self.val_path = os.path.join(data_path, "processed")
        self.test_path = os.path.join(data_path, "processed")
        self.batch_size = batch_size

    def __len__(self) -> int:
        """Return the length of the dataset."""
        len_train = len(os.listdir(self.train_path))
        len_val = len(os.listdir(self.val_path))
        len_test = len(os.listdir(self.test_path))
        return len_train + len_val + len_test

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return torch.load(os.path.join(self.train_path, f"{index}.pt"))

    def _get_num_labels(self) -> list[str | None]:
        """Return the list of unique labels from a CSV file."""
        df = pd.read_csv(os.path.join(self.data_path, "raw", "metadata.csv"))
        return len(df["label"].unique().tolist())

    def _get_train_loader(self) -> DataLoader:
        """Return a DataLoader for the training set."""
        train: torch.Tensor = torch.load(f"{self.train_path}\\train.pt", weights_only=True)
        train_img = train['images']
        train_labels = train['labels']
        train_dataset = TensorDataset(train_img, train_labels)
        return DataLoader(train_dataset, batch_size=self.batch_size)

    def _get_val_loader(self) -> DataLoader:
        """Return a DataLoader for the validation set."""
        val: torch.Tensor = torch.load(f"{self.val_path}\\val.pt", weights_only=True)
        val_img = val['images']
        val_labels = val['labels']
        val_dataset = TensorDataset(val_img, val_labels)
        return DataLoader(val_dataset, batch_size=self.batch_size)

    def _get_test_loader(self) -> DataLoader:
        """Return a DataLoader for the test set."""
        test: torch.Tensor = torch.load(f"{self.test_path}\\test.pt", weights_only=True)
        test_img = test['images']
        test_labels = test['labels']
        test_dataset = TensorDataset(test_img, test_labels)
        return DataLoader(test_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    typer.run(split_data_and_preprocess)

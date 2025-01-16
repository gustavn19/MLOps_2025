from pathlib import Path

import pandas as pd
import typer
from torch.utils.data import Dataset


class PokeData(Dataset):
    """A PyTorch Dataset for the Pokemon dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
    
    def get_labels(self) -> list[str | None]:
        """Return the list of unique labels from a CSV file."""
        df = pd.read_csv(self.data_path)
        
        return df["label"].unique().tolist()


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = PokeData(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)

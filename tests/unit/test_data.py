from torch.utils.data import Dataset

from data import PokeData


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = PokeData("data/raw")
    assert isinstance(dataset, Dataset)

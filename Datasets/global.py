from .CONFIG import CONFIG
from .Dataset import Dataset

def create_global_dataset():
    data = []
    for config in CONFIG:
        ds = Dataset(config=config)
        ds.__getitem__()

if __name__ == "__main__":
    create_global_dataset()

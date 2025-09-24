from pathlib import Path

class Dataset():
    def __init__(self, dataset_name: str, width: int, height: int):
        self.dataset_dir = Path(__file__).parent / "datasets" / dataset_name
        self.classes = self.create_class_encodings()
        self.width = width
        self.height = height

    def __get_item__(self):
        pass
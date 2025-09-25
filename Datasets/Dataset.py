from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import torch

@dataclass
class DatasetConfig:
    name: str
    width: int
    height: int
    classes: Dict[str, int]
    subfolders: List[str]

class Dataset():
    def __init__(self, config: DatasetConfig):
        self.dataset_dir = Path(__file__).parent / "datasets" / config['name']
        self.classes = config['classes']
        self.width = config['width']
        self.height = config['height']
        self.subfolders = config['subfolders']
        print(self.classes)

    def __getitem__(self):
        img_tensor = torch.tensor([])
        label_tensor = torch.tensor([])

        if "train" in self.subfolders:
            images_dir = Path(self.dataset_dir) / "train" / "images"
            labels_dir = Path(self.dataset_dir) / "train" / "labels"

            for image in (images_dir.iterdir()):
                
                # read label file into label tensor
                label_file = labels_dir / f"{image.stem}.txt"
                if not label_file.exists(): continue
                
                with open(label_file, "r") as file:
                    content = file.read().strip()
                    parts = content.split(" ")
                    if len(parts) != 5: continue
                    c, center_x, center_y, width, height = parts
                    if int(c) in self.classes.values():
                        print(c)


        return img_tensor, label_tensor

            
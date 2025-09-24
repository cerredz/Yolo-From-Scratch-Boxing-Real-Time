from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# A proper PyTorch Dataset should inherit from torch.utils.data.Dataset
class PascalVocDataset(Dataset):
    def __init__(self, train: bool = True, S: int = 7, B: int = 2, C: int = 20):
        dataset_path = "VOC2012_train_val/VOC2012_train_val" if train else "VOC2012_test/VOC2012_test"
        self.root = Path(__file__).parent.parent / "db" / dataset_path

        self.classes = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                        'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
                        'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13,
                        'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17,
                        'train': 18, 'tvmonitor': 19}
        
        self.S = S
        self.B = B
        self.C = C

        self.annotations_path = self.root / "Annotations"
        self.images_path = self.root / "JPEGImages"
        self.annotation_files = sorted(self.annotations_path.glob("*.xml"))
        
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, index):
        xml_path = self.annotation_files[index]
        image_path = self.images_path / f"{xml_path.stem}.jpg"
        
        img = Image.open(image_path).convert("RGB")
        original_width, original_height = img.size
        img_tensor = self.transform(img)

        target_label = torch.zeros(self.S, self.S, self.C + self.B * 5, dtype=torch.float32)
        
        root = ET.parse(xml_path).getroot()
        objects = root.findall("object")

        for obj in objects:
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            class_name = obj.find("name").text.replace(" ", "")

            center_x = (xmin + xmax) / 2 / original_width
            center_y = (ymin + ymax) / 2 / original_height
            width_rel = (xmax - xmin) / original_width
            height_rel = (ymax - ymin) / original_height

            i = int(self.S * center_y)
            j = int(self.S * center_x)

            x_cell = self.S * center_x - j
            y_cell = self.S * center_y - i

            if target_label[i, j, self.C] == 0:
                target_label[i, j, self.C] = 1.0

                target_label[i, j, self.C+1 : self.C+5] = torch.tensor(
                    [x_cell, y_cell, width_rel, height_rel], dtype=torch.float32
                )
                
                class_idx = self.classes[class_name]
                target_label[i, j, class_idx] = 1.0
        
        return img_tensor, target_label
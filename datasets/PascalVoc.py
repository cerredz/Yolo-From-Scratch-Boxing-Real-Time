from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
import torch

class PascalVocDataset():
    def __init__(self, train: bool = True, S: int = 7, B: int = 1, C:int = 20):
        dataset_path = Path("VOC2012_train_val") / "VOC2012_train_val" if train else Path("VOC2012_test") / "VOC2012_test"

        self.root = Path(__file__).parent.parent / "db" / dataset_path

        self.classes = {'aeroplane': 1,'bicycle': 2, 'bird': 3,'boat': 4,'bottle': 5,'bus': 6, 'car': 7,
            'cat': 8,'chair': 9,'cow': 10,'dining table': 11,'dog': 12,'horse': 13,'motorbike': 14,
            'person': 15,'potted plant': 16,'sheep': 17,'sofa': 18,'train': 19,
            'tv/monitor': 20
        }

        self.S = S
        self.B = B
        self.C = C

        self.data = self.load_dataset(train)

    def __resize_image(self, width, height):
        pass

    def __is_in_object(self, i, j, width, height, xmin, xmax, ymin, ymax):
        cell_width, cell_height = width // self.S, height // self.S

        cell_start_x, cell_end_x = i * cell_width, (i + 1) * cell_width
        cell_start_y, cell_end_y = j * cell_height, (j + 1) * cell_height

        if xmin >= cell_start_x and xmax <= cell_end_x and ymin >= cell_start_y and ymax <= cell_end_y:
            return True

        return False

    def __convert_xywh(self, i, j, width, height):
        cell_width, cell_height = width // self.S, height // self.S

        cell_start_x, cell_end_x = i * cell_width, (i + 1) * cell_width
        cell_start_y, cell_end_y = j * cell_height, (j + 1) * cell_height

        x = (cell_end_x - cell_start_x) // 2
        y = (cell_end_y - cell_end_y) // 2
        w = cell_width // 2
        h = cell_height // 2

        return x, y, w, h

    def __create_image_label(self, path: str, obj):
        ''' Split an image located at path into a SxS grid, where each cell is the pixel values '''
        img = Image.open(path)
        img = self.__resize_image(448, 448)
        width, height = img.size
        object_class = obj.find("name").text
        x_min, x_max = obj.find("bndbox").find("xmin").text, obj.find("bndbox").find("xmax").text
        y_min, y_max = obj.find("bndbox").find("ymin").text, obj.find("bndbox").find("ymax").text

        res = torch.zeros(self.S, self.S, self.B * 5 + self.C)
        
        for i in range(self.S):
            for j in range(self.S):
                is_in_obj = self.__is_in_object(i, j, width, height, x_min, x_max, y_min, y_max)
                x, y, w, h = self.__convert_xywh(i, j, width, height)
                
                res[i, j, self.C + 1:self.C + 5] = [x, y, w, h]

                if is_in_obj:
                    res[i, j, self.classes[object_class]] = 1

        return res

    def load_dataset(self, train: bool = True):
        annotations_path = Path(self.root) / "Annotations"
        images_path = Path(self.root) / "JPEGImages"
        
        annotation_files = sorted(annotations_path.glob("*.xml"))

        for file in annotation_files:
            image_path = images_path / f"{file.stem}.jpg"
            if not image_path.exists():
                print("Image file does not exist")
                continue

            root = ET.parse(file).getroot()
            first_obj = root.find("object")
            image_label = self.__create_image_label(image, first_obj)


            
            for obj in objects:
                name = obj.find("name").text
                y_min = obj.find("bndbox").find("ymin").text
                y_max = obj.find("bndbox").find("ymax").text
                
                

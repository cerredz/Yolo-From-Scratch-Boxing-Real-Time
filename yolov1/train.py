from datasets.PascalVoc import PascalVocDataset
from torch.utils.data import DataLoader
from pretrained.Yolo import Yolo
from torch.utils.data import DataLoader
import lightning as L
import torch
torch.set_float32_matmul_precision("medium")

def train(epochs: int, lr: int, batch_size: int):
    dataset = PascalVocDataset(train=True)
    yolo_model = Yolo(lr=lr)

    trainer = L.Trainer(max_epochs=EPOCHS)
    trainer.fit(model=yolo_model, train_dataloaders=DataLoader(dataset, batch_size=batch_size, shuffle=True))

if __name__ == "__main__":
    EPOCHS = 130
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3

    train(EPOCHS, LEARNING_RATE, BATCH_SIZE)
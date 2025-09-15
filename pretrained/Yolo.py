import torch # Added for the reshape/view function
from torch import nn
from lightning.pytorch import LightningModule

class Yolo(LightningModule):

    # define full Yolo nn architecture
    def __init__(self, S:int=7, B:int=2, C:int=20):
        super().__init__()

        self.S = S
        self.B = B
        self.C = C

        self.leaky_relu = nn.LeakyReLU(0.1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7,7), stride=2, padding=3),
            self.leaky_relu,
            nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )

        self.layer2= nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3,3), padding=1),
            self.leaky_relu,
            nn.MaxPool2d(kernel_size=(2,2),stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(1,1)), self.leaky_relu,
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1), self.leaky_relu,
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1)), self.leaky_relu,
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=1), self.leaky_relu,
            nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )

        self.layer4 = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1)), self.leaky_relu,
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=1), self.leaky_relu,
            # Block 2
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1)), self.leaky_relu,
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=1), self.leaky_relu,
            # Block 3
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1)), self.leaky_relu,
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=1), self.leaky_relu,
            # Block 4
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1)), self.leaky_relu,
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding=1), self.leaky_relu,
            # Rest of the block
            nn.Conv2d(in_channels=512, out_channels=512,kernel_size=(1,1)), self.leaky_relu,
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3), padding=1), self.leaky_relu,
            nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1,1)), self.leaky_relu,
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3), padding=1), self.leaky_relu,
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1,1)), self.leaky_relu,
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3), padding=1), self.leaky_relu,
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3,3), padding=1), self.leaky_relu,
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3,3), stride=2, padding=1), self.leaky_relu,
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3,3), padding=1), self.leaky_relu,
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3,3), padding=1), self.leaky_relu,
        )

        self.layer6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024 * S * S, out_features=4096), self.leaky_relu,
            nn.Linear(in_features=4096, out_features=S * S * (C + B * 5)),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        
        # FIXED: Correct reshape syntax, preserving the batch dimension
        x = x.view(-1, self.S, self.S, self.C + self.B * 5)
        return x
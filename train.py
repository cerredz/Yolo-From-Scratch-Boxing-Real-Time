from datasets.PascalVoc import PascalVocDataset

def train():
    pv_dataset = PascalVocDataset()

if __name__ == "__main__":
    EPOCHS = 130
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3

    train()
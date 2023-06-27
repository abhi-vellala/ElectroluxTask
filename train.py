from VGGModel import VGGNet
from loadData import LoadData
import numpy as np
import logging
import torch
import os
import pandas as pd
logging.basicConfig(format='[%(levelname)s] - %(message)s', level=logging.INFO)


BATCH_SIZE = 4
NUM_CLASSES = 10
EPOCHS = 2


torch.manual_seed(123)
train_path = "sample_data/train/"
test_path = "sample_data/test/"
dataloader = LoadData(train_path=train_path, test_path=test_path, transform=True, 
                      image_size=(224,224))

loader = dataloader.PrepareLoader(batch_size=BATCH_SIZE, shuffle=True)

model = VGGNet(dataloader=loader, num_classes=NUM_CLASSES)

res_data = pd.DataFrame(columns=["epoch", "train_loss", "train_accuracy", 
                                 "valid_loss", "valid_accuracy",
                                 "test_loss", "test_accuracy",])
idx = 0
for epoch in range(EPOCHS):
    logging.info(f"Epoch: {epoch+1} - TRAINING")
    train_loss, train_accuracies = model.Train()
    print(f"Train Loss: {np.mean(train_loss)}, Train Accuracy: {np.mean(train_accuracies)}")
    logging.info(f"Epoch: {epoch+1} - VALIDATION")
    valid_loss, valid_accuracies = model.Validate()
    print(f"Valid Loss: {np.mean(valid_loss)}, Valid Accuracy: {np.mean(valid_accuracies)}")
    logging.info(f"Epoch: {epoch+1} - TESTING")
    test_loss, test_accuracies = model.Test()
    print(f"Test Loss: {np.mean(test_loss)}, Test Accuracy: {np.mean(test_accuracies)}")
    model_save_path = os.path.join('model_checkpoints','epoch'+epoch+'_model.ckpt')
    model.saveModel()
    res_data.loc[idx, "epoch"] = epoch
    res_data.loc[idx, "train_loss"] = np.mean(train_loss)
    res_data.loc[idx, "train_accuracy"] = np.mean(train_accuracies)
    res_data.loc[idx, "valid_loss"] = np.mean(valid_loss)
    res_data.loc[idx, "valid_accuracies"] = np.mean(valid_accuracies)
    res_data.loc[idx, "test_loss"] = np.mean(test_loss)
    res_data.loc[idx, "test_accuracies"] = np.mean(test_accuracies)
    idx += 1
    
res_data.to_excel("results_data.xlsx", index=False)
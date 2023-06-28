from VGGModel import VGGNet
from loadData import LoadData
import numpy as np
import logging
import torch
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
logging.basicConfig(format='[%(levelname)s] - %(message)s', level=logging.INFO)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def sec_to_hours(seconds):
    a=str(int(seconds//3600))
    b=str(int((seconds%3600)//60))
    c=str(round((seconds%3600)%60, 4))
    d="{}:{}:{} seconds".format(a, b, c)
    return d


BATCH_SIZE = 16
NUM_CLASSES = 7
EPOCHS = 2


torch.manual_seed(123)
train_path = "sample_data/train/"
test_path = "sample_data/test/"
dataloader = LoadData(train_path=train_path, test_path=test_path, transform=True, 
                      image_size=(224,224))

loader = dataloader.PrepareLoader(batch_size=BATCH_SIZE, train_ratio=0.8, shuffle=True)

model = VGGNet(dataloader=loader, num_classes=NUM_CLASSES, pretrained=False)

res_data = pd.DataFrame(columns=["epoch", "train_loss", "train_accuracy", 
                                 "valid_loss", "valid_accuracy",
                                 "test_loss", "test_accuracy",])
idx = 0
start_time = time.time()
logging.info("Training start time recorded")
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
    model_save_path = os.path.join('model_checkpoints','epoch'+str(epoch)+'_model.ckpt')
    model.saveModel(model_save_path)
    res_data.loc[idx, "epoch"] = epoch
    res_data.loc[idx, "train_loss"] = np.mean(train_loss)
    res_data.loc[idx, "train_accuracy"] = np.mean(train_accuracies)
    res_data.loc[idx, "valid_loss"] = np.mean(valid_loss)
    res_data.loc[idx, "valid_accuracy"] = np.mean(valid_accuracies)
    res_data.loc[idx, "test_loss"] = np.mean(test_loss)
    res_data.loc[idx, "test_accuracy"] = np.mean(test_accuracies)
    idx += 1
    
res_data.to_excel("results_data.xlsx", index=False)
fig, axs = plt.subplots(1,2)
axs[0].plot(range(EPOCHS), res_data["train_loss"], label="Train Loss")
axs[0].plot(range(EPOCHS), res_data["valid_loss"], label="Valid Loss")
axs[0].plot(range(EPOCHS), res_data["test_loss"], label="Test Loss")
axs[0].set_title("Epochs Vs Losses")
axs[0].legend()

axs[1].plot(range(EPOCHS), res_data["train_accuracy"], label="Train Accuracy")
axs[1].plot(range(EPOCHS), res_data["valid_accuracy"], label="Valid Accuracy")
axs[1].plot(range(EPOCHS), res_data["test_accuracy"], label="Test Accuracy")
axs[1].set_title("Epochs Vs Accuracy")
axs[1].legend()

plt.savefig("model_result.png")

print(f"************* Model Executed in {sec_to_hours(time.time()-start_time)}*************")
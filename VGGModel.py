# train

import torch
import torchvision
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchmetrics.classification import Accuracy
from loadData import LoadData
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import logging
logging.basicConfig(format='[%(levelname)s] - %(message)s', level=logging.INFO)


class VGGNet:
    def __init__(self, dataloader, num_classes, pretrained=True, lr=0.001, momentum=0.9) -> None:
        torch.manual_seed(123)
        self.dataloader = dataloader
        self.num_classes = num_classes
        weights = VGG16_BN_Weights.DEFAULT if pretrained else None
        self.model = vgg16_bn(weights=weights, progress=True)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
  
    def Train(self):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        trainloader = self.dataloader['train']
        running_loss = 0.
        # start.record()
        train_losses = []
        accuracies = []
        for i, (inputs, labels) in enumerate(trainloader):
            # inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            train_losses.append(loss.item())
            preds = torch.argmax(outputs, 1)
            accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
            acc = accuracy(preds, labels).item()
            accuracies.append(acc)
            print(f"\t Batch: {i}/{len(trainloader)}")
            print(f"\t\t Current Loss: {round(loss.item(), 4)}, Running Loss: {round(running_loss, 4)}")
            print(f"\t\t Accuracy: {round(acc, 2)}")
            
        return train_losses, accuracies
            
        # end.record()
        # torch.cuda.synchronize()
        
    def Validate(self):
        validloader = self.dataloader['valid']
        val_losses = []
        val_accuracies = []
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(validloader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                loss = self.criterion(output, labels)
                val_losses.append(loss.item())
                preds = torch.argmax(output, dim=1)
                accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
                acc = accuracy(preds, labels).item()
                val_accuracies.append(acc)
                
                print(f"\t Batch: {i}/{len(validloader)}")
                print(f"\t\t Current Loss: {round(loss.item(),4)}")
                print(f"\t\t Accuracy: {round(acc,2)}")
                
        return val_losses, val_accuracies
    
    def Test(self):
        testloader = self.dataloader['test']
        test_losses = []
        test_accuracies = []
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(testloader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                loss = self.criterion(output, labels)
                test_losses.append(loss.item())
                preds = torch.argmax(output, dim=1)
                accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
                acc = accuracy(preds, labels).item()
                test_accuracies.append(acc)
                
                print(f"\t Batch: {i}/{len(testloader)}")
                print(f"\t\t Current Loss: {round(loss.item(),4)}")
                print(f"\t\t Accuracy: {round(acc,2)}")
                
        return test_losses, test_accuracies
        
                
                
        
        
    
            
            
            
            
        



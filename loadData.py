import os
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='[%(levelname)s] - %(message)s', level=logging.INFO)

np.random.seed(12345)

class LoadData:
    def __init__(self, train_path, test_path, transform=False, image_size=None) -> None:
        self.train_path = train_path
        self.test_path = test_path
        self.transform = transform
        if self.transform:
            self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        
        # self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        
    def TrainLoader(self):
        
        full_train_dataset = ImageFolder(root=self.train_path, transform=self.transform)
        logging.info(f"Number of training images Loaded: {len(full_train_dataset)}")
        train_dataset, valid_dataset = torch.utils.data.random_split(full_train_dataset, 
                                                                     [0.7,0.3])
        logging.info(f"Train-Validation Split ratio: (70, 30)")
        logging.info(f"Train set - {len(train_dataset)}, Validation Set - {len(valid_dataset)}")
        return train_dataset, valid_dataset
    
    def TestLoader(self):
        test_dataset = ImageFolder(root=self.test_path, transform=self.transform)
        logging.info(f"Number of testing images loaded: {len(test_dataset)}")
        return test_dataset
    
    def PrepareLoader(self, batch_size, shuffle):
        train_dataset, valid_dataset = self.TrainLoader()
        test_dataset = self.TestLoader()
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        logging.info("Data Loading Complete!")
        return {"train": trainloader, "valid": validloader, "test": testloader}
    
class TestDataLoader(LoadData):
    def __init__(self, train_path, test_path, transform=False, image_size=None) -> None:
        super().__init__(train_path, test_path, transform, image_size)
        # self.train_path = train_path
        
    def getCategories(self):
        for root, dirs, files in os.walk(self.train_path):
            if dirs:
                self.categories = sorted(dirs)
        return self.categories
            
    def makeImageMap(self, loader, batch_size):
        dataiter = iter(loader)
        images, labels = next(dataiter)
        img_map = {}
        while len(img_map.keys()) != len(self.categories):
            for i in range(batch_size):
                if not self.categories[labels[i]] in img_map.keys():
                    # hit+=1
                    # print(f"HIT: {hit}")
                    img_map[self.categories[labels[i]]] =  images[i]
                
            images, labels = next(dataiter)
            # cc += batch_size
            # print(cc)
            
        return img_map
    
    def plot_figs(self, imgs_dict, save_path, name):
        imgs = list(imgs_dict.values())
        labels = list(imgs_dict.keys())
        fig, axs = plt.subplots(2,8, figsize=(15,6))
        k = 0
        for i in range(2):
            for j in range(8):
                img = imgs[k]
                img = img / 2 + 0.5     # unnormalize
                npimg = img.numpy()
                axs[i,j].imshow(np.transpose(npimg, (1, 2, 0)))
                axs[i,j].set_title(labels[k])
                axs[i,j].axison = False
                # axs[i,j].axis("off")
                k += 1
        plt.savefig(os.path.join(save_path, str(name)+".png"))
        logging.info(f"Fig saved as: {os.path.join(save_path, str(name)+'.png')}")
        
        
        
    def makePlots(self, img_map, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        for i in range(0, len(img_map), 16):
            try:
                if i != 48:
                    self.plot_figs(dict(list(img_map.items())[i:i+16]), save_path, i)
                else:
                    self.plot_figs(dict(list(img_map.items())[i:]), save_path, i)
            except:
                pass
            
        
        
        
    
    
if __name__ == "__main__":
    train_path = "data/train/"
    test_path = "data/test/"
    dataloader = TestDataLoader(train_path=train_path, test_path=test_path, transform=True, 
                          image_size=(224,224))
    categories = dataloader.getCategories()
    batch_size = len(categories)
    loader = dataloader.PrepareLoader(batch_size=batch_size, shuffle=True)
    img_map = dataloader.makeImageMap(loader['train'], batch_size)
    dataloader.makePlots(img_map, "testing_imgs")
    
    # batch_size = 8
    # dataloader = LoadData(train_path=train_path, test_path=test_path, transform=True, 
    #                       image_size=(224,224))
    # loader = dataloader.PrepareLoader(batch_size=batch_size, shuffle=True, num_workers=2)
    
    # for root, dirs, files in os.walk(train_path):
    #     if dirs:
    #         classes = dirs
            
    # classes = sorted(classes)
    # batch_size = len(classes)
    # dataloader = LoadData(train_path=train_path, test_path=test_path, transform=True, 
    #                       image_size=(224,224))
    # loader = dataloader.PrepareLoader(batch_size=batch_size, shuffle=True, num_workers=2)
    
    # dataiter = iter(loader["train"])
    # images, labels = next(dataiter)
    # img_map = {}
    # cc = 0
    # hit = 0
    # while len(img_map.keys()) != len(classes):
    #     for i in range(batch_size):
    #         if not classes[labels[i]] in img_map.keys():
    #             hit+=1
    #             print(f"HIT: {hit}")
    #             img_map[classes[labels[i]]] =  images[i]
            
    #     images, labels = next(dataiter)
    #     cc += batch_size
    #     print(cc)
        
    # # print(img_map)
    # print(len(img_map.keys()))
    # def plot_figs(imgs_dict, name):
    #     imgs = list(imgs_dict.values())
    #     labels = list(imgs_dict.keys())
    #     fig, axs = plt.subplots(2,8, figsize=(15,6))
    #     k = 0
    #     for i in range(2):
    #         for j in range(8):
    #             img = imgs[k]
    #             img = img / 2 + 0.5     # unnormalize
    #             npimg = img.numpy()
    #             axs[i,j].imshow(np.transpose(npimg, (1, 2, 0)))
    #             axs[i,j].set_title(labels[k])
    #             axs[i,j].axison = False
    #             # axs[i,j].axis("off")
    #             k += 1
    #     plt.savefig(f"{name}.png")
    #     # plt.show()
        
    # for i in range(0, len(img_map), 16):
    #     try:
    #         if i != 48:
    #             plot_figs(dict(list(img_map.items())[i:i+16]),i)
    #         else:
    #             plot_figs(dict(list(img_map.items())[i:]),i)
    #     except:
    #         pass

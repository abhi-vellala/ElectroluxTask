import torch
from torchvision.models import vgg16_bn
from loadData import InferenceLoader
import os

img_path = "sample_data//test//capsule_faulty_imprint//000.png"

model_path = "model_checkpoints\epoch1_model.ckpt"

for root, dirs, files in os.walk("sample_data//train"):
        if dirs:
            categories = dirs
            
categories = sorted(categories)

model = vgg16_bn()
model.load_state_dict(torch.load(model_path))
# model = torch.load(model_path)
model.eval()
# print(model)
img = InferenceLoader(img_path, transform=True, image_size=(224,224)).imgGen()
output = model(img)
out_prob = torch.nn.Softmax(dim=-1)(output)
pred = torch.argmax(out_prob,1)
print(pred)
print(torch.max(out_prob))
print(categories[pred.item()])

print("loading assets...\n\n")

from collections import OrderedDict
from torchvision.transforms.functional import pil_to_tensor

from PIL import Image
import torch.nn as nn
import torch
import os

def load_model(model_path: str = "saved_model/model_weights-loss=0.03873.pth"):
    if not os.path.exists(model_path):
        print("model not found, exiting")
        exit()

    state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    model = nn.Sequential(
        nn.Conv2d(1, 128, 3),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2),
        nn.Dropout(),

        nn.Conv2d(128, 128, 3),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2),
        nn.Dropout(),

        nn.Flatten(),

        nn.Linear(3200, 128),
        nn.Dropout(),
        nn.Linear(128, 10),
    )

    model.load_state_dict(OrderedDict(
        zip(
            list(
                map(lambda s: s.lstrip('_orig_mod.'), list(state_dict.keys()))
            ),
            state_dict.values()
            
            )
        
    ))
    model.eval()
    return model

model = load_model()


print("The image path should be absolute, or if it is relative is should be based from project dir")
try:
    while True:
        print("or better, you can drag and drop the image below. press ctrl-c to exit.")
        image_path = str(input("Enter image path :")).strip()
        image = Image.open(image_path).resize((28, 28)).convert("L")

        pred = torch.argmax(model((pil_to_tensor(image)/255.0).expand(1, 1, 28, 28))).item()

        print(f"We think it is a {pred}")

        print("\n-------------\n\n")
    
except KeyboardInterrupt:
    print("")
    print("exiting...")
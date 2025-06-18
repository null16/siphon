import time
initial_start_time = time.time()

import os

def check_for_saved_model():

    if os.path.exists('saved_model'):

        saved_models = [p for p in os.listdir('saved_model') if p.startswith('model_weights-loss=0.')]

        if len(saved_models) > 0:
            print("Already saved model found.")
            for m in saved_models:
                print(f"'./saved_model/{m}'")

            confirmation = str(input("do you still want to train (Y/n): "))

            if confirmation in ['', 'y', 'Y']:
                return
            
            else:
                print("safely exiting, no changed made.")
                exit()


check_for_saved_model()

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn


import torch
device = torch.device("cpu")

torch.random.manual_seed(1602)
assert round(torch.randn(1).item(), 3) == round(-0.1493116319179535, 3), "seed is not set properly"

DATA_DIR = "data"
BATCH_SIZE = 256
EPOCHS = 10

training_data = MNIST(DATA_DIR, transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 10),

).to(device)

# model = torch.compile(model), does not work on some machines

optimizer = optim.Adam(model.parameters())
crossentropy = nn.CrossEntropyLoss()

loss_history = []

for epoch in range(EPOCHS):
    with tqdm(train_dataloader, desc=f"epoch: #{epoch}") as pbar:
        for i, (images, labels) in enumerate(pbar):
            optimizer.zero_grad()

            predictions = model(images)
            loss = crossentropy(predictions, labels)
            loss_history.append(loss.item())

            loss.backward()
            optimizer.step()

            pbar.set_postfix_str(f"loss: {loss:.5f}")

final_loss = sum(loss_history[-10:])/10

if final_loss > 0.1:
    print("warning: final loss is not low enough, it should be near 0.03 or atleast less than 0.1")

torch.save(model.state_dict(), f'saved_model/model_weights-loss={sum(loss_history[-10:])/10:.5f}.pth')

print("")
print(f"final loss: {final_loss:.5f}, model saved at './saved_model/model_weights-loss={sum(loss_history[-10:])/10:.5f}.pth'")
print(f"run in {time.time() - initial_start_time}s")

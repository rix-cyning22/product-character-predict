from torch.utils.data import DataLoader 
from models import ResidualTask
from utils import Dataset 
from torchvision.transforms import Normalize
from torch.nn.functional import mse_loss 
import torch
from time import time 
from torch.amp import autocast
import matplotlib.pyplot as plt

# torch._dynamo.config.capture_scalar_outputs = True
def moving_average_array(values, window_size):
    moving_averages = []
    for i in range(len(values)):
        if i + 1 < window_size:
            avg = sum(values[:i + 1]) / (i + 1)
        else:
            avg = sum(values[i + 1 - window_size:i + 1]) / window_size
        moving_averages.append(avg)
    return moving_averages


csvpath = "ml24/dataset/train_cleaned.csv"
img_dir = "train_images/processed"
channels_mean = [0.5, 0.5, 0.5]
channels_std = [0.2, 0.2, 0.2]
transforms = [Normalize(mean=channels_mean, std=channels_std)]

device = "cuda" if torch.cuda.is_available() else "cpu"

product_dataset = Dataset(csvpath=csvpath, img_dir=img_dir, extra_transforms=None, device="cuda")

dataloader = DataLoader(product_dataset, batch_size=32, shuffle=True)

# model = MTM()
model = ResidualTask()
model = torch.compile(model)
model = model.to(device)
    
num_batches = len(dataloader)
num_epochs = 15

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    start_time = time()
    running_loss = []
    print(f"Epoch {epoch+1} of {num_epochs}")
    for i, (img, mask, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        with autocast(device_type="cuda"):  
            preds = model(img, mask)
            loss = mse_loss(preds, labels)
            running_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            duration = (time() - start_time) / (i+1)
            print(f"[{i+1}/{num_batches}] Loss: {torch.mean(loss):.4f}; Time: {duration:.2f} s/batch")
        if i > 50:
            break
    moving_avg = moving_average_array(running_loss, window_size=5)
    plt.plot(running_loss)
    plt.plot(moving_avg)
    plt.show()

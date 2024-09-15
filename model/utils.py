import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision.transforms import ToTensor, Compose
import os

def convert_to_std_units(value, unit, entity):
    if entity in ["width", "height", "depth"]:
        return value * conversion_factors.get(unit, 1.0)
    elif entity in ["item_weight", "maximum_weight_recommendation"]:
        return value * conversion_factors.get(unit, 1.0)
    elif entity == "item_volume":
        return value * conversion_factors.get(unit, 1.0)
    elif entity == "voltage":
        return value * conversion_factors.get(unit, 1.0)
    elif entity == "wattage":
        return value * conversion_factors.get(unit, 1.0)
    return value

conversion_factors = {
    "centimetre": 0.01,
    "metre": 1.0,
    "millimetre": 0.001,
    "inch": 0.0254,
    "foot": 0.3048,
    "yard": 0.9144,
    "gram": 1e-3,
    "kilogram": 1.0,
    "microgram": 1e-9,
    "milligram": 1e-6,
    "ounce": 0.0283495,
    "pound": 0.453592,
    "ton": 1000.0,
    "gallon": 3.78541,
    "litre": 1.0,
    "millilitre": 1e-3,
    "cup": 0.24,
    "fluid ounce": 0.0295735,
    "volt": 1.0,
    "kilovolt": 1000.0,
    "millivolt": 1e-3,
    "watt": 1.0,
    "kilowatt": 1000.0,
}

task_info = {
    "height": ["metre", 0],
    "depth": ["metre", 1],
    "width": ["metre", 2],
    "item_weight": ["kilogram", 3],
    "maximum_weight_recommendation": ["kilogram", 4],
    "voltage": ["volt", 5],
    "item_volume": ["litre", 6],
    "wattage": ["watt", 7],
}

class Dataset(Dataset):
    def __init__(self, csvpath, img_dir, extra_transforms=None, sample=None, training=True, device="cpu"):
        super().__init__()
        self.df = pd.read_csv(csvpath)
        self.img_dir = img_dir 
        self.transform = [ToTensor()]
        if extra_transforms:
            self.transform += extra_transforms
        self.transform = Compose(self.transform)
        self.device = device
        if sample:
            self.df = self.df.sample(sample)
        self.training = training
    
    def __len__(self):
        return len(self.df)
    
    def prep_train_data(self, idx):
        names = self.df.iloc[idx, 0]
        imgpath = os.path.join(self.img_dir, names)
        img = Image.open(imgpath).convert("RGB")
        mask = torch.tensor(task_info[self.df.iloc[idx, 2]][1])
        label = torch.tensor(self.df.iloc[idx, 3])
        img = self.transform(img)
        return img, mask, label
    
    def prep_test_data(self, idx):
        names = self.df.iloc[idx, 1]
        names = os.path.basename(names)
        imgpath = os.path.join(self.img_dir, names)
        img = Image.open(imgpath).convert("RGB")
        mask = torch.tensor(task_info[self.df.iloc[idx, 2]][1])
        img = self.transform(img)
        return img, mask, None
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.training:
            img, mask, label = self.prep_train_data(idx)
            label = label.clone().detach().to(self.device, dtype=torch.float16)
        else:
            img, mask = self.prep_test_data(idx)    
        img = img.to(self.device, dtype=torch.float16)  
        mask = mask.to(self.device, dtype=torch.int16)
        if self.training:
            return img, mask, label 
        return img, mask
        
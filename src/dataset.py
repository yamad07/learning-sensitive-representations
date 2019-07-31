import torch
from torch.utils.data import Dataset
from skimage import io, color
from skimage.transform import resize
import glob
import os
import random


class SensitiveDataset(Dataset):

    def __init__(self, style_root_dir, content_root_dir, transform):
        self.style_file_list = glob.glob(os.path.join(style_root_dir, "*.jpg"))
        self.content_file_list = glob.glob(os.path.join(content_root_dir, "*.jpg"))
        self.transform = transform

    def __getitem__(self, index):
        source_file_name = random.choice(self.style_file_list)
        another_file_name = self.content_file_list[index]
        source_image = self.transform(
                resize(io.imread(source_file_name), (512, 512))).float()
        another_image = self.transform(
                resize(io.imread(another_file_name), (512, 512))).float()
        return source_image, another_image

    def __len__(self):
        return len(self.content_file_list)

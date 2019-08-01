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
        source_image = self._sampling_image(self.style_file_list)
        source_image = self.transform(source_image).float()

        another_image = self._sampling_image(self.content_file_list)
        another_image = self.transform(another_image).float()
        return source_image, another_image

    def __len__(self):
        return len(self.content_file_list)

    def _sampling_image(self, file_list):
        file_name = random.choice(file_list)
        image = resize(io.imread(file_name), (512, 512))
        if len(image.shape) != 3:
            image = self._sampling_image(file_list)
        return image

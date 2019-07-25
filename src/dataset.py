import torch
from torch.utils.data import Dataset


class SensitiveDataset(Dataset):

    def __init__(self, root_dir, transform):
        self.file_list = glob.glob(os.path.join(root_dir, "*.jpg"))
        self.transform = transform

    def __getitem__(self, index):
        source_file_name = self.file_list[index]
        another_file_name = self.file_list[random.choice(len(self.file_list))]
        source_image = self.transform(io.imread(file_name))
        another_image = self.transform(io.imread(file_name))
        return source_image, another_image

    def __len__(self):
        return len(self.file_list)

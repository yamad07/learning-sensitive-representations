from torch.utils.data import DataLoader
from src.model import LocalEncoder, SensitiveEncoder, Decoder
from src.trainer import Trainer
from src.dataset import SensitiveDataset

dataset = SensitiveDataset(
        root_dir="./images/"
        transform=transforms.Compose(
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            )
        )
dataloader = DataLoader(
        dataset
        )
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

trainer = Trainer(
        local_encoder=LocalEncoder(),
        sensitive_encoder=SensitiveEncoder(),
        decoder=Decoder(),
        train_data_loader=dataloader,
        )
trainer.train(100)

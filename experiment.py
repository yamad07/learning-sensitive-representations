from comet_ml import Experiment
from torch.utils.data import DataLoader
from torchvision import transforms
from src.model import Encoder, Decoder
from src.trainer import Trainer
from src.dataset import SensitiveDataset

experiment = Experiment(api_key="laHAJPKUmrD2TV2dIaOWFYGkQ",
                                project_name="learning-sensitive-features", workspace="yamad07")

dataset = SensitiveDataset(
        content_root_dir="./images/rkrkrk/",
        style_root_dir="./images/visualmemories_/",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )]
        )
        )
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

trainer = Trainer(
        encoder=Encoder(),
        decoder=Decoder(),
        train_data_loader=dataloader,
        experiment=experiment,
        )
trainer.train(1000)

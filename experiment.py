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
        style_root_dir="./images/mscoco/",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4109, 0.4062, 0.3984],
                [0.3176, 0.3150, 0.3169],
            )
        ])
    )
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

trainer = Trainer(
        encoder=Encoder(),
        decoder=Decoder(),
        train_data_loader=dataloader,
        experiment=experiment,
        weight_path="experiment/same-encoder"
        )
trainer.train(1000)

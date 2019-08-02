import torch
import torch.optim as optim
import torch.nn.functional as F
from .module import ada_in
import os


class Trainer(object):

    def __init__(
            self,
            encoder,
            decoder,
            train_data_loader,
            experiment,
            weight_path,
            alpha=1,
            beta=1,
            gamma=1,
            ):

        self.encoder = encoder
        self.decoder = decoder
        self.train_data_loader = train_data_loader

        self.encoder_optim = optim.Adam(
                self.encoder.parameters(), lr=0.0001)
        self.decoder_optim = optim.Adam(
                self.decoder.parameters(), lr=0.0001)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = torch.device("cuda:1")
        self.experiment = experiment
        self.weight_path = os.path.join("./weights/", weight_path)
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)

    def train(
            self,
            n_epochs,
            ):

        self.encoder.train().to(self.device)
        self.decoder.train().to(self.device)

        for epoch in range(n_epochs):
            for i, (style_images, content_images) in enumerate(
                    self.train_data_loader):

                style_images = style_images.to(self.device)
                content_images = content_images.to(self.device)

                all_content_features, content_features = self.encoder(content_images)
                decode_images = self.decoder(content_features)
                reconstruction_loss = F.mse_loss(content_images, decode_images)

                all_style_features, style_features = self.encoder(style_images)
                joint_features = ada_in(content_features, style_features)

                all_decode_joint_features, decode_joint_features = self.encoder(
                        self.decoder(joint_features))

                reconstruction_content_features_loss = F.mse_loss(
                        joint_features,
                        decode_joint_features,
                        )

                reconstruction_style_features_loss = 0
                for (decode_joint_features, style_features) in zip(
                        all_decode_joint_features, all_style_features):
                    reconstruction_style_features_loss += F.mse_loss(
                            style_features.mean((2, 3)),
                            decode_joint_features.mean((2, 3))) + \
                                    F.mse_loss(
                            style_features.view(
                                style_features.size(0),
                                style_features.size(1),
                                -1).var(2),
                            decode_joint_features.view(
                                style_features.size(0),
                                style_features.size(1),
                                -1).var(2),
                            )

                loss = self.alpha * reconstruction_content_features_loss + \
                        self.gamma * reconstruction_style_features_loss + self.beta * reconstruction_loss
                self.experiment.log_metric("reconstruction-loss", reconstruction_loss)
                self.experiment.log_metric("reconstruction-style-features-loss", reconstruction_style_features_loss)
                self.experiment.log_metric("reconstruction-content-features-loss", reconstruction_content_features_loss)
                self.experiment.log_metric("total", loss)
                self.encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()
                loss.backward()
                self.encoder_optim.step()
                self.decoder_optim.step()
            torch.save(self.encoder.state_dict(),
                    os.path.join(
                        self.weight_path, "Epoch_{}_local_encoder.pth".format(epoch)))
            torch.save(self.decoder.state_dict(),
                    os.path.join(
                        self.weight_path, "Epoch_{}_decoder.pth".format(epoch)))
            print("Epoch: {} ReconLoss: {} StyleFeatureLoss: {} TotalLoss: {}".format(
                epoch, reconstruction_loss, reconstruction_style_features_loss, loss))

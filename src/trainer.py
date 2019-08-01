import torch
import torch.optim as optim
import torch.nn.functional as F
from .module import ada_in
import os


class Trainer(object):

    def __init__(
            self,
            local_encoder,
            sensitive_encoder,
            decoder,
            train_data_loader,
            experiment,
            alpha=1.0,
            beta=1.0,
            delta=10.0,
            ):

        self.local_encoder = local_encoder
        self.sensitive_encoder = sensitive_encoder
        self.decoder = decoder
        self.train_data_loader = train_data_loader

        self.local_encoder_optim = optim.Adam(
                self.local_encoder.parameters(), lr=0.0001)
        self.sensitive_encoder_optim = optim.Adam(
                self.sensitive_encoder.parameters(), lr=0.0001)
        self.decoder_optim = optim.Adam(
                self.decoder.parameters(), lr=0.0001)
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.device = torch.device("cuda:0")
        self.experiment = experiment
        self.weight_path = "./weights/"

    def train(
            self,
            n_epochs,
            ):

        self.local_encoder.train().to(self.device)
        self.sensitive_encoder.train().to(self.device)
        self.decoder.train().to(self.device)

        for epoch in range(n_epochs):
            for i, (source_images, another_images) in enumerate(
                    self.train_data_loader):

                source_images = source_images.to(self.device)
                another_images = another_images.to(self.device)

                local_source_features = self.local_encoder(source_images)
                sensitive_source_features = self.sensitive_encoder(source_images)
                joint_features = torch.mul(local_source_features, sensitive_source_features)
                decode_images = self.decoder(joint_features)
                reconstruction_loss = F.mse_loss(source_images, decode_images)

                local_another_features = self.local_encoder(another_images)
                sensitive_another_features = self.sensitive_encoder(another_images)

                transferd_style_another_features = ada_in(
                        sensitive_another_features,
                        sensitive_source_features,
                        )
                joint_features = torch.mul(local_source_features, transferd_style_another_features)

                decode_sensitive_source_features = self.sensitive_encoder(
                        self.decoder(joint_features))

                adain_style_loss = F.mse_loss(
                        sensitive_source_features.view(
                        sensitive_source_features.size(0),
                        sensitive_source_features.size(1),
                        -1,
                        ).mean(2),
                        decode_sensitive_source_features.view(
                        sensitive_source_features.size(0),
                        sensitive_source_features.size(1),
                        -1
                        ).mean(2)) +  F.mse_loss(
                                sensitive_source_features.view(
                                    sensitive_source_features.size(0),
                                    sensitive_source_features.size(1),
                                    -1).var(2),
                                decode_sensitive_source_features.view(
                                    sensitive_source_features.size(0),
                                    sensitive_source_features.size(1),
                                    -1).var(2),
                                )
                reconstruction_features_loss = F.mse_loss(
                        transferd_style_another_features,
                        decode_sensitive_source_features)

                loss = self.alpha * reconstruction_loss + \
                        self.beta * reconstruction_features_loss + \
                        self.delta * adain_style_loss
                self.experiment.log_metric("reconstruction-loss", reconstruction_loss)
                self.experiment.log_metric("reconstruction-features-loss", reconstruction_features_loss)
                self.experiment.log_metric("adain-style-loss", adain_style_loss)
                self.experiment.log_metric("total", loss)
                self.local_encoder_optim.zero_grad()
                self.sensitive_encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()
                loss.backward()
                self.local_encoder_optim.step()
                self.sensitive_encoder_optim.step()
                self.decoder_optim.step()
            torch.save(self.sensitive_encoder.state_dict(),
                    os.path.join(
                        self.weight_path, "Epoch_{}_sensitive_encoder.pth".format(epoch)))
            torch.save(self.local_encoder.state_dict(),
                    os.path.join(
                        self.weight_path, "Epoch_{}_local_encoder.pth".format(epoch)))
            torch.save(self.decoder.state_dict(),
                    os.path.join(
                        self.weight_path, "Epoch_{}_decoder.pth".format(epoch)))
            print("Epoch: {} ReconLoss: {} SensitiveFeatureLoss: {} TotalLoss: {}".format(
                epoch, reconstruction_loss, reconstruction_features_loss, loss))

import torch.optim as optim


class Trainer(object):

    def __init__(
            self,
            local_encoder,
            sensitive_encoder,
            decoder,
            train_data_loader,
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

    def train(
            self,
            n_epochs,
            ):

        self.local_encoder.train()
        self.sensitive_encoder.train()

        for epoch in range(n_epochs):
            for i, (source_images, another_images) in enumerate(
                    self.train_data_loader):

                local_source_features = self.local_encoder(source_images)
                sensitive_source_feautes = self.sensitive_encoder(
                        source_images)
                sensitive_features = F.interpolate(
                        sensitive_features, size=(
                            local_source_features.size(2),
                            local_source_features.size(3),
                            ))
                joint_features = torch.mul(
                        local_source_features,
                        sensitive_source_features)
                decode_images = self.decoder(joint_features)

                reconstruction_loss = F.mse_loss(source_images, decode_images)

                local_another_features = self.local_encoder(another_images)
                joint_features = torch.mul(
                        local_another_features,
                        sensitive_source_features)

                decode_sensitive_source_features = self.sensitive_encoder(
                        self.decoder(joint_features))
                reconstruction_features_loss = F.mse_loss(
                        sensitive_source_features,
                        decode_sensitive_source_features)

                loss = alpha * reconstruction_loss + (1 - alpha) * reconstruction_features_loss
                self.local_encoder_optim.zero_grad()
                self.sensitive_encoder_optim.zero_grad()
                self.decoder_optim.zero_grad()
                loss.backward()
                self.local_encoder_optim.step()
                self.sensitive_encoder_optim.step()
                self.decoder_optim.step()
            print(loss)

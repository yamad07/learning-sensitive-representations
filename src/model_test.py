from model import LocalEncoder
from model import SensitiveEncoder
from model import Decoder
import torch
import torch.nn.functional as F


if __name__ == "__main__":
    local_encoder = LocalEncoder()
    sensitive_encoder = SensitiveEncoder()
    decoder = Decoder()

    x = torch.randn(1, 3, 224, 224)
    local_features = local_encoder(x)
    sensitive_features = sensitive_encoder(x)
    sensitive_features = F.interpolate(sensitive_features, size=(14, 14))
    joint_features = torch.mul(local_features, sensitive_features)
    print(joint_features.size())

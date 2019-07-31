import torch

def ada_in(content, style):
    style_mean = style.mean(
            (2, 3)).unsqueeze(2).unsqueeze(3).expand(
                    tuple(content.size()))
    style_var = style.view(
            style.size(0), style.size(1), -1
            ).var(2).unsqueeze(2).unsqueeze(3).expand(
                    tuple(content.size()))

    return torch.mul(content, style_var) + style_mean

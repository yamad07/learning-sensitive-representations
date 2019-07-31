import torch

def ada_in(content, style):
    style_mean = style.mean(
            (2, 3)).unsqueeze(2).unsqueeze(3).expand(
                    tuple(content.size()))
    style_var = style.view(
            style.size(0), style.size(1), -1
            ).var(2).unsqueeze(2).unsqueeze(3).expand(
                    tuple(content.size()))
    content_mean = content.mean(
            (2, 3)).unsqueeze(2).unsqueeze(3).expand(
                    tuple(content.size()))
    content_var = content.view(
            content.size(0), content.size(1), -1
            ).var(2).unsqueeze(2).unsqueeze(3).expand(
                    tuple(content.size()))
    return torch.mul((content - content_mean) / content_var, style_var) + style_mean

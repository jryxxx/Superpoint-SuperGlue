import torch
import torch.nn as nn


class MaxPoolNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=1)


if __name__ == "__main__":
    model = MaxPoolNet()
    input = torch.randn(1, 3, 512, 512)
    onnx_filename = "./output/norm.onnx"
    torch.onnx.export(
        model,
        input,
        onnx_filename,
        opset_version=13
    )

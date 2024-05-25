#!/usr/bin/env python

import argparse
import os

import numpy as np
import onnx
import onnxruntime
import torch

from models.superpoint import SuperPoint

from onnxconverter_common import float16

def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def main():
    parser = argparse.ArgumentParser(
        description="script to convert superpoint model from pytorch to onnx"
    )
    parser.add_argument(
        "--weight_file",
        default="models/weights/superpoint_v1.pth",
        help="pytorch weight file (.pth)",
    )
    parser.add_argument(
        "--output_dir", default="output", help="onnx model file output directory"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    weight_file = args.weight_file

    # load model
    superpoint_model = SuperPoint()
    pytorch_total_params = sum(p.numel() for p in superpoint_model.parameters())
    print("total number ff params: ", pytorch_total_params)

    # initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    superpoint_model.load_state_dict(torch.load(weight_file, map_location=map_location))
    superpoint_model.eval()

    # create input to the model for onnx trace
    input = torch.randn(1, 1, 240, 320)

    torch_out = superpoint_model(input)
    onnx_filename = os.path.join(
        output_dir, weight_file.split("/")[-1].split(".")[0] + ".onnx"
    )

    # export the model
    torch.onnx.export(
        superpoint_model,  # model being run
        input,  # model input (or a tuple for multiple inputs)
        onnx_filename,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=13,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model input names
        output_names=["scores", "descriptors"],  # the model output names
    )

    # onnx_model = onnx.load_model(onnx_filename)
    # trans_model = float16.convert_float_to_float16(onnx_model, min_positive_val=1e-7, max_finite_val=1e4, keep_io_types=False,
    #                      disable_shape_infer=False, op_block_list=None, node_block_list=None)
    # onnx.save_model(trans_model, "output/superpoint_v1_fp16.onnx")


if __name__ == "__main__":
    main()

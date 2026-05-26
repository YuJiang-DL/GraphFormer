# -*- coding: utf-8 -*-

import os
import sys
import argparse
from train import Train
import torch
import copy

def Parser_main():
    parser = argparse.ArgumentParser(description="Deep cox analysis model")
    parser.add_argument("--DatasetType", default="TCGA", help="TCGA_BRCA or BORAME or BORAME_Meta or BORAME_Prog",
                        type=str)
    parser.add_argument("--learning_rate", default=0.0001, help="Learning rate", type=float)
    parser.add_argument("--weight_decay", default=0.00005, help="Weight decay rate", type=float)
    parser.add_argument("--clip_grad_norm_value", default=2.0, help="Gradient clipping value", type=float)
    parser.add_argument("--batch_size", default=4, help="batch size", type=int)
    parser.add_argument("--num_epochs", default=50, help="2", type=int)
    parser.add_argument("--dropedge_rate", default=0.25, help="Dropedge rate for GAT", type=float)
    parser.add_argument("--dropout_rate", default=0.25, help="Dropout rate for MLP", type=float)
    parser.add_argument("--graph_dropout_rate", default=0.25, help="Node/Edge feature dropout rate", type=float)
    parser.add_argument("--initial_dim", default=100, help="Initial dimension for the GAT", type=int)
    parser.add_argument("--attention_head_num", default=2, help="Number of attention heads for GAT", type=int)
    parser.add_argument("--number_of_layers", default=5, help="Whole number of layer of GAT", type=int)
    parser.add_argument("--FF_number", default=3, help="Selecting set for the five fold cross validation", type=int)
    parser.add_argument("--model", default="MPGAT", help="GAT_custom/DeepGraphConv_TCGA_STAD/PatchGCN_TCGA_STAD/GIN/MIL/MIL-attention", type=str)
    parser.add_argument("--gpu", default=0, help="Target gpu for calculating loss value", type=int)
    parser.add_argument("--norm_type", default="layer", help="BatchNorm=batch/LayerNorm=layer", type=str)
    parser.add_argument("--MLP_layernum", default=3, help="Number of layers for pre/pose-MLP", type=int)
    parser.add_argument("--with_distance", default="Y", help="Y/N; Including positional information as edge feature", type=str)
    parser.add_argument("--simple_distance", default="N", help="Y/N; Whether multiplying or embedding positional information", type=str)
    parser.add_argument("--loss_type", default="PRELU", help="RELU/Leaky/PRELU", type=str)
    parser.add_argument("--residual_connection", default="Y", help="Y/N", type=str)


    return parser.parse_args()

def main():
    Argument = Parser_main()

    best_model, checkpoint_dir, fig_dir, bestepoch = Train(Argument)
    #Analyze(Argument, best_model, checkpoint_dir, fig_dir, bestepoch, best_select="Y")


def run_multiple_experiments():
    initial_arguments = Parser_main()  # Get the initial arguments

    # Loop over a list of modified arguments or configurations
    for i in range(5):  # For example, 5 experiments
        # Create a copy of the initial arguments and modify it for the next experiment
        new_arguments = copy.deepcopy(initial_arguments)

        # Modify the arguments as per your requirement (e.g., change learning rate, dataset, etc.)
        new_arguments.FF_number = 0 + i

        print(
            f"Running experiment {i + 1} with model: {new_arguments.model} and cv: {new_arguments.FF_number}")

        Train(new_arguments)  # Run the main function with modified arguments


def run_multiple_experiments2():
    initial_arguments = Parser_main()  # Get the initial arguments
    initial_arguments.model = "TransGCN"
    # Loop over a list of modified arguments or configurations
    for i in range(5):  # For example, 5 experiments
        # Create a copy of the initial arguments and modify it for the next experiment
        new_arguments = copy.deepcopy(initial_arguments)

        # Modify the arguments as per your requirement (e.g., change learning rate, dataset, etc.)
        new_arguments.FF_number = 0 + i

        print(
            f"Running experiment {i + 1} with model: {new_arguments.model} and cv: {new_arguments.FF_number}")

        Train(new_arguments)  # Run the main function with modified arguments

if __name__ == "__main__":

    # run_multiple_experiments()  # Execute multiple experiments automatically
    run_multiple_experiments()  # Execute multiple experiments automatically
    # main()


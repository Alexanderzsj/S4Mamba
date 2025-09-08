import matplotlib.pyplot as plt
import numpy as np
import os


def plot_loss(train_loss_list, val_loss_list, save_path="loss_ima.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Training Loss', color='blue')
    plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_acc(val_acc_list, save_path="acc_img.png"):
    """
    Plot the validation accuracy curve and save it to a file.
    
    :param val_acc_list: List of validation accuracy values (e.g., OA) per epoch
    :param save_path: Path to save the plot image
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_acc_list) + 1), val_acc_list, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Overall Accuracy (OA)')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
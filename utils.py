"""
Script that includes some helper functions.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from typing import List, Dict
import numpy as np


def visualize_class_distributions(data_frame: pd.core.frame.DataFrame, class_column: str):
    """Plot the distribution of the class column from a pandas DataFrame.

    Args:
        data_frame (pd.core.frame.DataFrame): The DataFrame containing the data.
        class_column (str): The column name representing the class to be visualized.
      
    """
    class_counts = data_frame[class_column].value_counts()

    plt.figure(figsize=(10,5))
    
    sns.barplot(x=class_counts.index, y=class_counts.values, palette=["g", "b"])
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(range(len(class_counts.index)), class_counts.index)
    plt.title('Class Distributions', fontsize=15)
    
    plt.show()


def visualize_feature_distributions(data_frame: pd.core.frame.DataFrame, features: list):
    """Plot the distribution of specified features from a pandas DataFrame using seaborn.

    Args:
        data_frame (pd.core.frame.DataFrame): The DataFrame containing the data.
        features (list): A list of column names (strings) representing the features to be visualized.

    """
    # Calculate the number of rows needed for subplots
    n = len(features)
    rows = n // 2
    rows += n % 2

    # Set up the matplotlib figure
    fig, axes = plt.subplots(rows, 2, figsize=(15, 7 * rows))
    axes = axes.ravel()  # axes are 2-dimensional so we unravel them

    # Set seaborn style for nicer graphics
    sns.set(style="whitegrid")

    # Plot each feature
    for i, feature in enumerate(features):
        ax = axes[i]
        sns.histplot(data_frame[feature], kde=True, ax=ax, bins=30, color='blue', linewidth=0)
        ax.set_title(f'Distribution of {feature}', fontsize=15)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)

    # If the number of features is odd, we need to remove the last empty subplot
    if len(features) % 2 != 0:
        fig.delaxes(axes[-1])

    fig.tight_layout()
    plt.show()

def visualize_results(results: Dict[str, List[float]]):
    """
    Visualizes the training and testing loss and accuracy over epochs.

    Args:
        results (Dict[str, List[float]]): A dictionary containing lists of metrics. 
                                          The keys are "train_loss", "train_acc", 
                                          "test_loss", and "test_acc".

    """
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))  # 1 row, 2 columns
    num_epochs = len(results['train_loss'])
    # Plot losses
    axs[0].plot(results['train_loss'], label='Training Loss')
    axs[0].plot(results['test_loss'], label='Testing Loss')
    axs[0].set_title('Training and Testing Loss Over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].set_xticks(np.arange(0, num_epochs, 1))

    # Plot accuracies
    axs[1].plot(results['train_acc'], label='Training Accuracy')
    axs[1].plot(results['test_acc'], label='Testing Accuracy')
    axs[1].set_title('Training and Testing Accuracy Over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].set_xticks(np.arange(0, num_epochs, 1))


    # Display the plot
    plt.show()

def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculates the accuracy of predictions compared to true values.

    This function compares each element of `y_true` to its corresponding
    element in `y_pred` to determine if they are equal. It then sums the 
    number of equal elements, divides by the total number of elements, 
    and multiplies by 100 to get a percentage representing the accuracy 
    of the predictions.

    Args:
        y_true (torch.Tensor): A tensor of ground truth (actual) values.
        y_pred (torch.Tensor): A tensor of predicted values.

    Returns:
        The percentage of correct predictions, i.e. the accuracy.
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
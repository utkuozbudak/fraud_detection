import torch
import pandas
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split


def create_dataloaders(X: pandas.DataFrame, 
                       y: pandas.DataFrame, 
                       batch_size: int = 32, 
                       shuffle: bool = True,
                       dtype: torch.dtype = torch.float32,
                       sampling: str = "undersampling",
                       sampling_strategy: str = 'majority',
                       test_size: float = 0.2) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    This function takes input and target data, applies a sampling strategy and 
    splits the data into training and testing datasets. These datasets are then 
    converted into PyTorch DataLoaders for use in a training loop.
    
    Args:
        X (pandas.DataFrame): Input data.
        y (pandas.DataFrame): Target data.
        batch_size (int, optional): Size of the batches. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data in the DataLoader. Defaults to True.
        dtype (torch.dtype, optional): The desired data type for the input data. Defaults to torch.float32.
        sampling (str, optional): The sampling method to use. Options are "undersampling" and "smote". Defaults to "undersampling".
        sampling_strategy (str, optional): The sampling strategy for the chosen method. Defaults to 'majority'.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
    
    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: A tuple containing the training and testing DataLoaders.
    
    Raises:
        ValueError: If an unrecognized sampling strategy is passed.
    """
    
    # Undersampling
    if sampling == "undersampling":
        undersample = RandomUnderSampler(sampling_strategy=sampling_strategy)
        X_over, y_over = undersample.fit_resample(X, y)
        print(f"Undersampling applied\nNew class counts: {Counter(y_over)}")
    # SMOTE
    elif sampling == "smote":
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_over, y_over = smote.fit_resample(X, y)
        print(f"SMOTE applied\n New class counts: {Counter(y_over)}")
    else:
        raise ValueError(f"Sampling strategy '{sampling}' not recognized. Choose 'undersampling' or 'smote'.")

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_over,
                                                        y_over,
                                                        random_state=42,
                                                        test_size=test_size)
    
    # Print information about the data
    print(f"Shape of X_train: {X_train.shape} | Length of X_train: {len(X_train)}")
    print(f"Shape of y_train: {y_train.shape} | Length of y_train: {len(y_train)}")
    print(f"Shape of X_test: {X_test.shape} | Length of X_test: {len(X_test)}")
    print(f"Shape of y_test: {y_test.shape} | Length of y_test: {len(y_test)}")
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=dtype)
    y_train_tensor = torch.tensor(y_train.values, dtype=dtype)
    X_test_tensor = torch.tensor(X_test.values, dtype=dtype)
    y_test_tensor = torch.tensor(y_test.values, dtype=dtype)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader
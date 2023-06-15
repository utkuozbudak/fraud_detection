"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from tqdm.auto import tqdm
from typing import Tuple, Dict, List
import torch.optim.lr_scheduler


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for one epoch.

    Args:
        model (torch.nn.Module): PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): A DataLoader object to iterate over.
        loss_fn (torch.nn.Module): A PyTorch loss function to minimize.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to apply.
        device (torch.device): A target device to train model on (e.g. "cuda" or "cpu").

    Returns:
        Tuple[float, float]: A tuple of training loss and training accuracy metrics,
        in the form of (train_loss, train_acc).
    """
    
    # Model into training mode
    model.train()
    
    # Initialize loss and accuracy
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to device
        X, y = X.to(device), y.to(device)
        
        # 1. Forward pass
        y_logits = model(X).squeeze()
        y_preds = torch.round(torch.sigmoid(y_logits))
        
        # 2. Compute loss & accuracy
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()
        train_acc += (y_preds == y).sum().item() / len(y_preds)
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        
        # 4. Backward pass
        loss.backward()
        
        # 5. Optimizer step
        optimizer.step()
    
    # Compute average loss and accuracy
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss, train_acc
    
    
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for one epoch.

    Args:
        model (torch.nn.Module): A PyTorch model to be tested.
        dataloader (torch.utils.data.DataLoader): A DataLoader object for the model to be tested on.
        loss_fn (torch.nn.Module): A PyTorch loss function to compute loss on test data.
        device (torch.device): A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        Tuple[float, float]: A tuple of testing loss and testing accuracy metrics,
        in the form of (test_loss, test_acc).
    """
    # Model into evaluation mode
    model.eval()
    
    # Initialize test loss and test accuracy
    test_loss, test_acc = 0, 0
    # Inference context manager
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # Send data to device
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_logits = model(X).squeeze()
            test_preds = torch.round(torch.sigmoid(test_logits))
            
            # 2. Calculate and accumulate loss
            loss = loss_fn(test_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_acc += (test_preds == y).sum().item() / len(test_preds)
    
    # Compute average loss and accuracy per batch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          scheduler: torch.optim.lr_scheduler) -> Dict[str, List]:
    
    """ Trains ans tests a PyTorch model.
    Calculates, prints and stores evaluation metrics throughout.
    
    Args:
        model (torch.nn.Module): A PyTorch model to be trained and tested.
        train_dataloader (torch.utils.data.DataLoader): A DataLoader object for the model to be trained on.
        test_dataloader (torch.utils.data.DataLoader): A DataLoader object for the model to be tested on.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer to apply.
        loss_fn (torch.nn.Module): A PyTorch loss function to calculate loss.
        epochs (int): Number of epochs to train the model for.
        device (torch.device): A target device to train and test on (e.g. "cuda" or "cpu").
        scheduler ()
        
    Returns:
        Dict[str, List]: A dictionary of training and testing metrics.
    """
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        
        scheduler.step()
        
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        
        # Print out results
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
    return results
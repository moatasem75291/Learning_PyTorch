
from typing import Tuple, List, Dict

from tqdm.auto import tqdm
import torch

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    
    """
    Performs a single training step for one epoch.

    Args:
        model (torch.nn.Module): The neural network model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the training data in batches.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
        loss_fn (torch.nn.Module): The loss function used to calculate the error between predictions and targets.
        device (torch.device): The device (CPU or GPU) on which the computations are performed.

    Returns:
        Tuple[float, float]: A tuple containing:
            - train_acc (float): The average training accuracy across all batches.
            - train_loss (float): The average training loss across all batches.

    Example usage:
        train_acc, train_loss = train_step(model, train_dataloader, optimizer, loss_fn, device)
    """
    
    model.train()
    train_loss, train_acc = 0, 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # 1. Forward Pass
        output = model(X).to(device)# outputs will be a logits
        
        # 2. Calculate the loss
        loss = loss_fn(output, y)
        train_loss += loss.item() # loss per batch
        
        #3. Optimizer zero grad
        optimizer.zero_grad()
        
        # 4. backward pass
        loss.backward()
        
        # 5. Optimizer step
        optimizer.step()
        
        # Calculate the acc
        y_pred_classes = torch.argmax(torch.softmax(output, dim=-1), dim=-1)
        train_acc += ((y_pred_classes == y).sum() / len(y)).item() # acc per batch
        
#     adjust metrics to get all loss/accuracy values per all of the data
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_acc, train_loss


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluates the model on a validation or test dataset.

    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the validation or test data in batches.
        loss_fn (torch.nn.Module): The loss function used to calculate the error between predictions and targets.
        device (torch.device): The device (CPU or GPU) on which the computations are performed.

    Returns:
        Tuple[float, float]: A tuple containing:
            - test_acc (float): The average accuracy across all batches in the test dataset.
            - test_loss (float): The average loss across all batches in the test dataset.

    Example usage:
        test_acc, test_loss = test_step(model, test_dataloader, loss_fn, device)
    """
    
    model.eval()
    test_loss, test_acc = 0, 0
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            
            X, y = X.to(device), y.to(device)
            outputs = model(X)

            loss = loss_fn(outputs, y)
            
            test_loss += loss.item()
            
            y_pred_classes = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            test_acc += (y_pred_classes == y).sum().item() / len(y)
            
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    
    return test_acc, test_loss


    def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: torch.device
    ) -> Dict[str, List[float]]:
        """
        Trains and evaluates the model over a specified number of epochs.
        
        Args:
            model (torch.nn.Module): The neural network model to train and evaluate.
            train_dataloader (torch.utils.data.DataLoader): DataLoader providing the training data in batches.
            test_dataloader (torch.utils.data.DataLoader): DataLoader providing the test data in batches.
            loss_fn (torch.nn.Module): The loss function used to calculate the error between predictions and targets.
            optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
            epochs (int, optional): The number of epochs to train the model. Defaults to 5.
            device (torch.device, optional): The device (CPU or GPU) on which the computations are performed. Defaults to the global `device`.
        
        Returns:
            Dict[str, List[float]]: A dictionary containing lists of accuracy and loss values for both the training and test datasets across all epochs:
                - "train_acc" (List[float]): Training accuracy for each epoch.
                - "train_loss" (List[float]): Training loss for each epoch.
                - "test_acc" (List[float]): Test accuracy for each epoch.
                - "test_loss" (List[float]): Test loss for each epoch.
        
        Example usage:
            results = train(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs=10, device=device)
        """
        #     Result dictionary
        result = {
            "train_acc": [],
            "train_loss": [],
            "test_acc": [],
            "test_loss": [],
        }
        #     loop throught train, and test step functions
        for epoch in tqdm(range(epochs)):
            
            train_acc, train_loss = train_step(model, train_dataloader, optimizer, loss_fn, device)
            
            test_acc, test_loss = test_step(model, test_dataloader, loss_fn, device)
            
        #         print out the result
            print(
                f"Epoch: {epoch} | Train acc: {train_acc:.4f}, Train Loss: {train_loss:.4f} | Test acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}"
            )
        
            result["train_acc"].append(train_acc)
            result["train_loss"].append(train_loss)
            result["test_acc"].append(test_acc)
            result["test_loss"].append(test_loss)
        
            ###New: Experiment Tracking ###
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                },
                global_step=epoch
            )
        
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={
                    "train_acc": train_acc,
                    "test_acc": test_acc
                },
                global_step=epoch
            )
            writer.add_graph(
                model=model,
                input_to_model=torch.rand(32, 3, 224, 224).to(device)
            )
        
        # Close the writer
        writer.close()
        
        ###End New ###
        return result
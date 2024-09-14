from pathlib import Path
import torch

def save_model(
    model: torch.nn.Module,
    target_dir: str,
    model_name:str
):
    """
    Saves the model's state dictionary to a specified directory.

    Args:
        model (torch.nn.Module): The neural network model to save.
        target_dir (str): The target directory where the model will be saved.
        model_name (str): The name of the model file. Must end with '.pth' or '.pt'.

    Raises:
        AssertionError: If `model_name` does not end with '.pth' or '.pt'.

    Returns:
        None

    Example usage:
        save_model(model, target_dir="models/", model_name="my_model.pth")
    """
    
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    assert model_name.endswith("pth") or model_name.endswith("pt"), "model_name parameter should end with `pth` or `pt`."
    model_path = target_dir_path / model_name
    
    torch.save(model.state_dict(), model_path)
    
    print(f"The model was saved successfully in: {model_path}")


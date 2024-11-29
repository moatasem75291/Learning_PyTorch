import multiprocessing

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# NUM_WORKERS = multiprocessing.cpu_count()

def create_dataloader(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
#     NUM_WORKERS: int = NUM_WORKERS,
):
    """
    Creates DataLoader objects for training and testing datasets.

    Args:
        train_dir (str): Path to the directory containing training images.
        test_dir (str): Path to the directory containing testing images.
        transform (transforms.Compose): A torchvision transforms.Compose object containing the transformations to apply to the images.
        NUM_WORKERS (int, optional): Number of worker processes to use for data loading. Defaults to the number of available CPU cores.
        batch_size (int): Number of images to process in each batch.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training dataset.
            - test_loader (DataLoader): DataLoader for the testing dataset.
            - class_names (list): List of class names found in the training directory.

    Example usage:
        train_dir = "/path/to/train"
        test_dir = "/path/to/test"
        transform = transforms.Compose([...])
        batch_size = 32
        
        train_loader, test_loader, class_names = create_dataloader(train_dir, test_dir, transform, batch_size=batch_size)
    """
    # Read the image data from the Folders
    train_data = datasets.ImageFolder(root=train_dir, transform=transform, )
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)
    
    class_names = train_data.classes
    
    # Put the data into loaders
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
#         num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
#         num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, test_loader, class_names

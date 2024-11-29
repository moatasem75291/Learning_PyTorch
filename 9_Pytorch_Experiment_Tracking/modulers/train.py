import os
from timeit import default_timer as timer

import torch
from torchvision import transforms

import data_setup, model_builder, engine, utils

# Some Hyper-parameters
NUM_EPOCHS = 50
HIDDEN_UNITS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# SET THE DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories
train_dir = 'data/pizza_steak_sushi/train'
test_dir = 'data/pizza_steak_sushi/test'
TARGET_MODEDL_PATH = "modulers/model/"

# Create a tansform
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create our data loaders
train_loader, test_loader, class_names = data_setup.create_dataloader(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create a model
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Setup loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

# Start time
start = timer()

# Start training with engine file
result = engine.train(
    model=model,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=device
)

# End the timer
end = timer()

print(f"Total training time: {end - start:.4f} seconds")

# Save the model
utils.save_model(
    model,
    TARGET_MODEDL_PATH,
    "going_moduler_script_mode_tinyvgg_model.pth"
)

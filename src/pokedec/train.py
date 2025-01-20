import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
import typer
from model import get_model
from torch. profiler import ProfilerActivity, profile, record_function, tensorboard_trace_handler
from tqdm import tqdm

import wandb
from data import PokeData

logger = logging.getLogger(__name__)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def train_model(num_classes: int = 1000,
                batch_size: int = 32,
                num_epochs: int = 100,
                lr: float = 1e-4,
                wd: float = 1e-4,
                use_wandb: bool = False,
                profiling: bool = False,
                export_model: bool = False
                ) -> None:
    '''
    Trains a model to classify Pokemon using the specified hyperparameters.

    Args:
        num_classes (int): The number of output classes for the classifier.
        batch_size (int): The number of samples in each batch.
        num_epochs (int): The number of epochs to train the model.
        lr (float): The learning rate for the optimizer.
        wd (float): The weight decay for the optimizer.
        use_wandb (bool): Whether to use Weights & Biases for logging.
        profiling (bool): Whether to enable profiling during training.
        export_model (bool): Whether to export the model to ONNX format after training.

    Returns:
        None: The function performs training, validation, and artifact logging but does not return any value.
    '''

    # Initialize Weights & Biases
    if use_wandb:
        run = wandb.init(
            project="pokedec_train",
            entity="pokedec_mlops",
            config={"lr": lr, "batch_size": batch_size, "epochs": num_epochs},
            job_type="train",
            name=f"pokedec_model_bs_{batch_size}_e_{num_epochs}_lr_{lr}_wd_{wd}",
        )

    # Load model
    model = get_model(num_classes=num_classes)
    logger.info(f"Using device: {DEVICE}")
    model = model.to(DEVICE)

    # Load data
    poke_data = PokeData(data_path='data', batch_size=batch_size, num_workers=1)
    train_loader = poke_data._get_train_loader()
    val_loader = poke_data._get_val_loader()

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)   


    # Training loop
    for epoch in tqdm(range(num_epochs)):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        if profiling:
            # Profiling context manager
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True, 
                        with_stack=True,
                        on_trace_ready=tensorboard_trace_handler("models/profiler"),
                        ) as prof:

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()

                    # Forward pass
                    with record_function("model_forward"):
                        outputs = model(inputs)
                    with record_function("loss_computation"):
                        loss = criterion(outputs, labels)
                    with record_function("backward_pass"):
                        loss.backward()
                    with record_function("optimizer_step"):
                        optimizer.step()

                    prof.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

        else:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # Log epoch statistics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        logger.info(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")


        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if use_wandb:
            wandb.log({"train_loss": epoch_loss, "train_accuracy": epoch_acc, "val_loss": val_loss, "val_accuracy": val_acc})

        # Step the scheduler
        scheduler.step()

    logger.info("Finished Training")


    # Save the model
    if use_wandb:
        # Sweep
        #os.makedirs("models/sweep", exist_ok=True)
        #torch.save(model.state_dict(), f"models/sweep/pokedec_model_bs_{batch_size}_e_{num_epochs}_lr_{lr}_wd_{wd}.pth")

        # Single model
        torch.save(model.state_dict(), f"models/pokedec_model_bs_{batch_size}_e_{num_epochs}_lr_{lr}_wd_{wd}_best.pth")
        
        artifact = wandb.Artifact(
            name=f"pokedec_models_best",
            type="model",
            description="Model trained to classfiy Pokemon during sweep",
        )
        artifact.add_file(f"models/pokedec_model_bs_{batch_size}_e_{num_epochs}_lr_{lr}_wd_{wd}_best.pth")
        run.log_artifact(artifact)
        
        wandb.finish()

    # Export model to ONNX format
    if export_model:
        model.eval()
        img, target = next(iter(val_loader))
        img, target = img.to(DEVICE), target.to(DEVICE)
        
        # Choose the first image in the batch
        img = img[0].unsqueeze(0)

        torch.onnx.export(
            model,
            img,
            "models/model.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
        )


if __name__ == "__main__":
    typer.run(train_model)

import os
import torch.nn as nn
import timm
import torch
import typer
import wandb

from data import PokeData


def load_model(model_path: str):
    model = timm.create_model('resnet50d.ra4_e3600_r224_in1k', pretrained=False)
    
    model.load_state_dict(torch.load(model_path))
    
    model.eval()

    return model

def evaluate(model_version: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load trained model
    run = wandb.init()
    artifact = run.use_artifact(f'pokedec_mlops/pokedec_mlops/pokedec_model:v{model_version}', type='model')
    artifact_dir = artifact.download()
    model_path = os.path.join(artifact_dir, 'resnet50d_finetuned.pth')
    print(artifact_dir)
    model = timm.create_model('resnet50d.ra4_e3600_r224_in1k', pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Load test data
    poke_data = PokeData(data_path='data', batch_size=32)
    test_loader = poke_data._get_test_loader()

    # Define loss
    criterion = nn.CrossEntropyLoss()

    # Initialize metrics
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    # Evaluation loop
    with torch.no_grad():
        for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = test_correct / test_total
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    return test_acc

if __name__ == "__main__":
    typer.run(evaluate)
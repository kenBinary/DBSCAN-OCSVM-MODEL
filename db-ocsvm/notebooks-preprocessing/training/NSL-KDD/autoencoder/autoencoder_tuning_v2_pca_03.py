#############################################################
# Gives the best autoencoder configuration based on         #
#  reconstruction error of normal, anomaly, or difference   #
# Also exports the best models to ONNX format               #
#############################################################
import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from models import BatchNormAutoencoder
from tuning_utils import train_autoencoderV2
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Ensure output directory exists
os.makedirs("tuning_results", exist_ok=True)
results_file = f"tuning_results/autoencoder_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}_pca.log"

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configuration parameters
sample_size = 0.01
use_sample = True
epochs = 10
validation_criterion = "difference"  # Options: "normal", "anomaly", "difference"

# Model configurations
# hidden_dims_list = [[54, 36], [40, 25, 17], [33, 20], [56, 32, 16], [50, 40]]
# latent_dim_list = [30, 8, 10, 4, 38]

hidden_dims_list = [[54, 36]]
latent_dim_list = [30]

# Training parameters
lr_list = [0.1, 0.01, 0.001, 0.0001]
batch_size_list = [32, 64, 128, 256]


# Function to export model to ONNX format
def export_to_onnx(model, input_dim, file_path):
    model.eval()
    # Create dummy input tensor for ONNX export
    dummy_input = torch.randn(1, input_dim, device=device)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        file_path,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=17,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )
    print(f"Model exported to ONNX: {file_path}")


# Load and prepare training data
train_set_path = (
    "/home/jbct/Projects/thesis/db-ocsvm/data/processed/NSL-KDD/train_set_full.csv"
)
train_df = pd.read_csv(train_set_path)

if use_sample:
    train_df = train_df.sample(frac=sample_size, random_state=42).reset_index(drop=True)

print(f"Training data shape: {train_df.shape}")

X_train_full = train_df.values

# Split data into train and validation sets
X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=42)

print("Performing PCA")
pca = PCA(n_components=68)
X_train = pca.fit_transform(X_train.values)
X_val = pca.transform(X_val.values)


# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_val_tensor = torch.FloatTensor(X_val)

# Create datasets
train_dataset = TensorDataset(X_train_tensor)
val_dataset = TensorDataset(X_val_tensor)

print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

# Get input dimension
input_dim = X_train.shape[1]

# Load test data
test_set_path = (
    "/home/jbct/Projects/thesis/db-ocsvm/data/processed/NSL-KDD/test_set.csv"
)
test_df = pd.read_csv(test_set_path)
print(f"Test data shape: {test_df.shape}")

# Split test data into features and labels
X_test = test_df.drop(
    columns=["attack_binary", "attack_categorical", "attack_class"]
).values
y_test = test_df["attack_binary"].values

print("Performing PCA on test data")
X_test = pca.transform(X_test)

# Separate normal and anomaly samples from test set
X_test_normal = X_test[y_test == 1]
X_test_anomaly = X_test[y_test == -1]

print(f"Normal test samples: {X_test_normal.shape[0]}")
print(f"Anomaly test samples: {X_test_anomaly.shape[0]}")

# Convert test data to PyTorch tensors
X_test_normal_tensor = torch.FloatTensor(X_test_normal).to(device)
X_test_anomaly_tensor = torch.FloatTensor(X_test_anomaly).to(device)

# Create DataLoaders for test data evaluation
normal_test_dataset = TensorDataset(X_test_normal_tensor)
anomaly_test_dataset = TensorDataset(X_test_anomaly_tensor)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=256, shuffle=False)
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=256, shuffle=False)


# Function to calculate reconstruction error
def calculate_reconstruction_error(model, loader):
    model.eval()
    total_loss = 0
    total_samples = 0
    criterion = nn.MSELoss(reduction="none")

    with torch.no_grad():
        for batch in loader:
            x = batch[0]
            outputs = model(x)
            # Calculate MSE for each sample
            loss = criterion(outputs, x)
            loss = loss.mean(dim=1)
            total_loss += torch.sum(loss).item()
            total_samples += x.size(0)

    return total_loss / total_samples


# Function to evaluate a model's reconstruction performance
def evaluate_model(model):
    normal_loss = calculate_reconstruction_error(model, normal_test_loader)
    anomaly_loss = calculate_reconstruction_error(model, anomaly_test_loader)
    loss_difference = anomaly_loss - normal_loss

    return {
        "normal_loss": normal_loss,
        "anomaly_loss": anomaly_loss,
        "loss_difference": loss_difference,
    }


# Dictionary to store best results for each model configuration
best_results = {}

# Ensure ONNX export directory exists
os.makedirs("onnx_models", exist_ok=True)

# Main grid search loop
with open(results_file, "w") as f:
    f.write("AUTOENCODER TUNING RESULTS\n")
    f.write("==========================\n\n")
    f.write(f"Validation criterion: {validation_criterion}\n\n")

    # Iterate through each model configuration
    for i, (hidden_dims, latent_dim) in enumerate(
        zip(hidden_dims_list, latent_dim_list)
    ):
        model_id = f"Model_{i+1}"
        model_name = f"{model_id}_hidden{hidden_dims}_latent{latent_dim}"

        if validation_criterion == "normal":
            best_metric = float("inf")  # Lower is better for normal reconstruction
        elif validation_criterion == "anomaly":
            best_metric = -float("inf")  # Higher is better for anomaly reconstruction
        elif validation_criterion == "difference":
            best_metric = -float("inf")  # Higher difference is better

        best_config = None
        best_model_path = None
        best_model_state = None

        f.write(f"[{model_name}]\n")
        f.write(f"Hidden dimensions: {hidden_dims}\n")
        f.write(f"Latent dimension: {latent_dim}\n\n")
        f.write("Training parameters tested:\n")

        # Iterate through all combinations of learning rates and batch sizes
        for lr in lr_list:
            for batch_size in batch_size_list:
                print(f"\nTraining {model_name} with lr={lr}, batch_size={batch_size}")

                # Create model
                autoencoder = BatchNormAutoencoder(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    latent_dim=latent_dim,
                    activation_type="LeakyReLU",
                    output_activation_type="Sigmoid",
                ).to(device)

                # Create data loaders
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False
                )

                # Setup optimizer
                optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
                criterion = nn.MSELoss()

                # Define model path for training
                current_model_path = (
                    f"best_models/autoencoder_{model_id}_lr{lr}_bs{batch_size}.pth"
                )
                os.makedirs("best_models", exist_ok=True)

                # Train autoencoder using the existing utility function
                # Including its internal validation
                print(f"Starting training autoencoder")
                history = train_autoencoderV2(
                    model=autoencoder,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    val_loader=val_loader,
                    epochs=epochs,
                    best_model_path=current_model_path,
                    verbose=True,
                )
                print(f"Training complete")

                # Load the best model from training
                checkpoint = torch.load(current_model_path)
                autoencoder.load_state_dict(checkpoint["model_state_dict"])
                best_val_loss = checkpoint.get("val_loss", float("inf"))

                # Evaluate model on test data
                print("Evaluating model on test set")
                eval_metrics = evaluate_model(autoencoder)
                normal_loss = eval_metrics["normal_loss"]
                anomaly_loss = eval_metrics["anomaly_loss"]
                loss_difference = eval_metrics["loss_difference"]

                print(f"Normal reconstruction loss: {normal_loss:.6f}")
                print(f"Anomaly reconstruction loss: {anomaly_loss:.6f}")
                print(f"Difference (anomaly - normal): {loss_difference:.6f}")

                # Log results
                f.write(f"  lr={lr}, batch_size={batch_size}:\n")
                f.write(f"    Normal reconstruction loss: {normal_loss:.6f}\n")
                f.write(f"    Anomaly reconstruction loss: {anomaly_loss:.6f}\n")
                f.write(f"    Difference: {loss_difference:.6f}\n")

                # Choose metric based on validation criterion
                if validation_criterion == "normal":
                    current_metric = normal_loss
                    is_better = current_metric < best_metric
                elif validation_criterion == "anomaly":
                    current_metric = anomaly_loss
                    is_better = current_metric > best_metric
                else:  # difference
                    current_metric = loss_difference
                    is_better = current_metric > best_metric

                # Track best configuration for this model
                if is_better:
                    best_metric = current_metric
                    best_config = {
                        "lr": lr,
                        "batch_size": batch_size,
                        "val_loss": best_val_loss,
                        "normal_loss": normal_loss,
                        "anomaly_loss": anomaly_loss,
                        "loss_difference": loss_difference,
                    }
                    best_model_path = current_model_path
                    best_model_state = checkpoint["model_state_dict"]

        # Copy the best model for this configuration to a dedicated location and export to ONNX
        if best_model_path:
            # Save PyTorch model
            best_final_path = f"best_models/autoencoder_{model_id}_pca.pth"
            os.makedirs("best_models", exist_ok=True)
            import shutil

            shutil.copy(best_model_path, best_final_path)
            print(f"Best model saved to {best_final_path}")

            # Create a fresh model instance with the best architecture
            best_autoencoder = BatchNormAutoencoder(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
                activation_type="LeakyReLU",
                output_activation_type="Sigmoid",
            ).to(device)

            # Load the best model state
            best_autoencoder.load_state_dict(best_model_state)

            # Export to ONNX
            onnx_path = f"onnx_models/autoencoder_{model_id}_pca.onnx"
            export_to_onnx(best_autoencoder, input_dim, onnx_path)

            # Add ONNX path to the configuration info
            best_config["onnx_path"] = onnx_path

        # Save best config for this model
        best_results[model_name] = best_config

        # Log best configuration for this model
        f.write("\nBEST CONFIGURATION:\n")
        f.write(f"Training validation loss: {best_config['val_loss']:.6f}\n")
        f.write(f"Normal reconstruction loss: {best_config['normal_loss']:.6f}\n")
        f.write(f"Anomaly reconstruction loss: {best_config['anomaly_loss']:.6f}\n")
        f.write(f"Difference: {best_config['loss_difference']:.6f}\n")
        f.write(f"Learning rate: {best_config['lr']}\n")
        f.write(f"Batch size: {best_config['batch_size']}\n")
        f.write(f"ONNX model path: {best_config.get('onnx_path', 'Not exported')}\n")
        f.write("\n" + "=" * 50 + "\n\n")

# Print overall best results
print(
    f"\nBEST RESULTS FOR EACH MODEL CONFIGURATION (criterion: {validation_criterion}):"
)
for model_name, config in best_results.items():
    if validation_criterion == "normal":
        print(
            f"{model_name}: normal_loss={config['normal_loss']:.6f}, lr={config['lr']}, batch_size={config['batch_size']}, ONNX: {config.get('onnx_path', 'Not exported')}"
        )
    elif validation_criterion == "anomaly":
        print(
            f"{model_name}: anomaly_loss={config['anomaly_loss']:.6f}, lr={config['lr']}, batch_size={config['batch_size']}, ONNX: {config.get('onnx_path', 'Not exported')}"
        )
    else:
        print(
            f"{model_name}: loss_difference={config['loss_difference']:.6f}, lr={config['lr']}, batch_size={config['batch_size']}, ONNX: {config.get('onnx_path', 'Not exported')}"
        )

print(f"\nResults saved to: {results_file}")
print("ONNX models exported to 'onnx_models/' directory")

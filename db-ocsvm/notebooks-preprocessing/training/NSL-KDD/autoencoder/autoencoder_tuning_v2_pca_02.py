####################################################
# Gives the best autoencoder configuration based on#
#  the lowest validation loss from the train set   #
####################################################
from sklearn.decomposition import PCA
import torch
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from models import BatchNormAutoencoder
from tuning_utils import train_autoencoderV2
import os
from datetime import datetime

# Import ONNX
import torch.onnx

# Ensure output directories exist
os.makedirs("tuning_results", exist_ok=True)
os.makedirs("best_models", exist_ok=True)
os.makedirs("best_models/onnx", exist_ok=True)

results_file = f"tuning_results/autoencoder_pca_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configuration parameters
sample_size = 0.01
use_sample = True
epochs = 10

# Model configurations
# hidden_dims_list = [[54, 36], [40, 25, 17], [33, 20], [56, 32, 16], [50, 40]]
# latent_dim_list = [30, 8, 10, 4, 38]

hidden_dims_list = [[54, 36]]
latent_dim_list = [30]

# Training parameters
lr_list = [0.1, 0.01, 0.001, 0.0001]
batch_size_list = [32, 64, 128, 256]

# Load and prepare data
train_set_path = (
    "/home/jbct/Projects/thesis/db-ocsvm/data/processed/NSL-KDD/train_set_full.csv"
)
train_df = pd.read_csv(train_set_path)

if use_sample:
    train_df = train_df.sample(frac=sample_size, random_state=42).reset_index(drop=True)

print(f"Training data shape: {train_df.shape}")

# Split data into train and validation sets
X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=42)
print("Performing PCA")
pca = PCA(n_components=68)
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)

print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_val_tensor = torch.FloatTensor(X_val)

# Create datasets
train_dataset = TensorDataset(X_train_tensor)
val_dataset = TensorDataset(X_val_tensor)

# Get input dimension
input_dim = X_train.shape[1]


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


# Dictionary to store best results for each model configuration
best_results = {}

# Main grid search loop
with open(results_file, "w") as f:
    f.write("AUTOENCODER TUNING RESULTS\n")
    f.write("==========================\n\n")

    # Iterate through each model configuration
    for i, (hidden_dims, latent_dim) in enumerate(
        zip(hidden_dims_list, latent_dim_list)
    ):
        model_id = f"Model_pca_{i+1}"
        model_name = f"{model_id}_pca_hidden{hidden_dims}_latent{latent_dim}"
        best_val_loss = float("inf")  # Initialize with infinity to find minimum
        best_config = None
        best_model = None

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

                # Define model path
                best_model_path = (
                    f"best_models/autoencoder_pca_{model_id}_lr{lr}_bs{batch_size}.pth"
                )

                print(f"Starting training autoencoder")
                # Train autoencoder
                history, is_good_model = train_autoencoderV2(
                    model=autoencoder,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    val_loader=val_loader,
                    epochs=epochs,
                    best_model_path=best_model_path,
                    verbose=True,
                )
                print(f"Training complete")

                # Load best model from training
                checkpoint = torch.load(best_model_path)
                autoencoder.load_state_dict(checkpoint["model_state_dict"])
                final_val_loss = checkpoint.get("val_loss", float("inf"))

                print(f"Validation Loss: {final_val_loss:.6f}")

                f.write(
                    f"  lr={lr}, batch_size={batch_size}: val_loss={final_val_loss:.6f}\n"
                )

                # Track best configuration for this model based on validation loss
                if final_val_loss < best_val_loss:
                    best_val_loss = final_val_loss
                    best_config = {
                        "lr": lr,
                        "batch_size": batch_size,
                        "val_loss": final_val_loss,
                        "model_path": f"best_models/autoencoder_pca_{model_name}_best.pth",
                        "onnx_path": f"best_models/onnx/autoencoder_pca_{model_name}_best.onnx",
                    }

                    # Save the best model for this configuration
                    torch.save(checkpoint, best_config["model_path"])
                    best_model = autoencoder  # Keep reference to the best model

        # Save best config for this model
        best_results[model_name] = best_config

        # Export the best model to ONNX
        if best_model is not None:
            export_to_onnx(best_model, input_dim, best_config["onnx_path"])

        # Log best configuration for this model
        f.write("\nBEST CONFIGURATION (lowest validation loss):\n")
        f.write(f"Validation Loss: {best_config['val_loss']:.6f}\n")
        f.write(f"Learning rate: {best_config['lr']}\n")
        f.write(f"Batch size: {best_config['batch_size']}\n")
        f.write(f"Model saved at: {best_config['model_path']}\n")
        f.write(f"ONNX model saved at: {best_config['onnx_path']}\n")
        f.write("\n" + "=" * 50 + "\n\n")

# Print overall best results
print("\nBEST RESULTS FOR EACH MODEL CONFIGURATION (lowest validation loss):")
for model_name, config in best_results.items():
    print(
        f"{model_name}: val_loss={config['val_loss']:.6f}, lr={config['lr']}, batch_size={config['batch_size']}"
    )
    print(f"  - PyTorch model: {config['model_path']}")
    print(f"  - ONNX model: {config['onnx_path']}")

print(f"\nResults saved to: {results_file}")

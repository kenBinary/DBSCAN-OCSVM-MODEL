import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from models import BatchNormAutoencoder
from tuning_utils import train_autoencoderV2
import os
from datetime import datetime

# Ensure output directory exists
os.makedirs("tuning_results", exist_ok=True)
results_file = (
    f"tuning_results/autoencoder_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configuration parameters
sample_size = 0.05
use_sample = True
epochs = 10

# Model configurations
# hidden_dims_list = [[96, 64], [96, 72, 48, 32], [100, 64, 32], [116, 96, 64, 48]]
# latent_dim_list = [55, 16, 8, 24]

hidden_dims_list = [[96, 64]]
latent_dim_list = [55]

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

X_train_full = train_df.values

# Split data into train and validation sets
X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=42)
X_train = X_train.values
X_val = X_val.values

print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_val_tensor = torch.FloatTensor(X_val)

# Create datasets
train_dataset = TensorDataset(X_train_tensor)
val_dataset = TensorDataset(X_val_tensor)

# Get input dimension
input_dim = X_train.shape[1]

# Load test data
test_set_path = (
    "/home/jbct/Projects/thesis/db-ocsvm/data/processed/NSL-KDD/test_set.csv"
)
test_df = pd.read_csv(test_set_path)
print(f"Test data shape: {test_df.shape}")

# Splitting test data into X and y
X_test = test_df.drop(
    columns=["attack_binary", "attack_categorical", "attack_class"]
).values
y_test = test_df["attack_binary"].values
y_test_class = test_df["attack_class"]

# Prepare test data for encoding
X_test_tensor = torch.FloatTensor(X_test).to(device)
test_dataset = TensorDataset(X_test_tensor)
X_test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# Function to evaluate a model
def evaluate_model(autoencoder, batch_size):
    autoencoder.eval()

    # Create data loaders for training and evaluation
    X_train_full_tensor = torch.FloatTensor(X_train_full).to(device)
    train_full_dataset = TensorDataset(X_train_full_tensor)
    train_full_loader = DataLoader(train_full_dataset, batch_size=256, shuffle=False)

    # Encode training data
    X_train_encoded = []
    with torch.no_grad():
        for data in train_full_loader:
            data_x = data[0].to(device)
            encoded = autoencoder.encode(data_x)
            X_train_encoded.append(encoded.cpu().numpy())

    X_train_encoded = np.vstack(X_train_encoded)

    # Train OCSVM on encoded features
    ocsvm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.2)
    ocsvm.fit(X_train_encoded)

    # Encode test data
    X_test_encoded = []
    with torch.no_grad():
        for data in X_test_loader:
            data_x = data[0].to(device)
            encoded = autoencoder.encode(data_x)
            X_test_encoded.append(encoded.cpu().numpy())

    X_test_encoded = np.vstack(X_test_encoded)

    # Make predictions
    y_pred = ocsvm.predict(X_test_encoded)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=-1)
    recall = recall_score(y_test, y_pred, pos_label=-1)
    f1 = f1_score(y_test, y_pred, pos_label=-1)

    return acc, precision, recall, f1


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
        model_id = f"Model_{i+1}"
        model_name = f"{model_id}_hidden{hidden_dims}_latent{latent_dim}"
        best_acc = 0
        best_config = None

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
                    f"best_models/autoencoder_{model_id}_lr{lr}_bs{batch_size}.pth"
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

                # Evaluate model
                print("Training and evaluting OCSVM model")
                acc, precision, recall, f1 = evaluate_model(autoencoder, batch_size)
                print(
                    f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
                )
                print("Finished evaluating model")

                f.write(f"  lr={lr}, batch_size={batch_size}: accuracy={acc:.4f}\n")

                # Track best configuration for this model
                if acc > best_acc:
                    best_acc = acc
                    best_config = {
                        "lr": lr,
                        "batch_size": batch_size,
                        "accuracy": acc,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }

        # Save best config for this model
        best_results[model_name] = best_config

        # Log best configuration for this model
        f.write("\nBEST CONFIGURATION:\n")
        f.write(f"Accuracy: {best_config['accuracy']:.4f}\n")
        f.write(f"Precision: {best_config['precision']:.4f}\n")
        f.write(f"Recall: {best_config['recall']:.4f}\n")
        f.write(f"F1 Score: {best_config['f1']:.4f}\n")
        f.write(f"Learning rate: {best_config['lr']}\n")
        f.write(f"Batch size: {best_config['batch_size']}\n")
        f.write("\n" + "=" * 50 + "\n\n")

# Print overall best results
print("\nBEST RESULTS FOR EACH MODEL CONFIGURATION:")
for model_name, config in best_results.items():
    print(
        f"{model_name}: accuracy={config['accuracy']:.4f}, lr={config['lr']}, batch_size={config['batch_size']}"
    )

print(f"\nResults saved to: {results_file}")

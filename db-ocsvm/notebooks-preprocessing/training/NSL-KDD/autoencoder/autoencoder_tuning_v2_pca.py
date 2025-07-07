import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from models import BatchNormAutoencoder
from tuning_utils import train_autoencoderV2
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
import os
import json
from datetime import datetime

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
sample_size = 0.01
use_sample = True
epochs = 50

# Model configurations
hidden_dims_list = [[54, 36], [40, 25, 17], [33, 20], [56, 32, 16], [50, 40]]
latent_dim_list = [30, 8, 10, 4, 38]

# Training parameters
lr_list = [0.1, 0.01, 0.001, 0.0001]
batch_size_list = [32, 64, 128, 256]

# Create results directory if it doesn't exist
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Create a log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(results_dir, f"autoencoder_tuning_results_{timestamp}.txt")

# Load and prepare data
train_set_path = (
    "/home/jbct/Projects/thesis/db-ocsvm/data/processed/NSL-KDD/train_set_full.csv"
)
train_df = pd.read_csv(train_set_path)

if use_sample:
    train_df = train_df.sample(frac=sample_size, random_state=42).reset_index(drop=True)

print(f"Training data shape: {train_df.shape}")

X_train_full = train_df.values

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

# Create a dictionary to store the best results for each model configuration
best_results = {}

# Loop through each model configuration
for config_idx, (hidden_dims, latent_dim) in enumerate(
    zip(hidden_dims_list, latent_dim_list)
):
    config_name = f"Config_{config_idx+1}_hidden{hidden_dims}_latent{latent_dim}"
    print(f"\n{'-'*50}")
    print(f"Testing model configuration: {config_name}")
    print(f"{'-'*50}")

    best_accuracy = -1
    best_params = None

    # Loop through each learning rate and batch size combination
    for lr in lr_list:
        for batch_size in batch_size_list:
            print(f"\nTesting with lr={lr}, batch_size={batch_size}")

            # Create a unique model path for this configuration
            best_model_path = f"best_models/autoencoder_config{config_idx+1}_lr{lr}_bs{batch_size}.pth"
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

            # Prepare data splits for this batch size
            X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=42)
            X_train = X_train.values
            X_val = X_val.values

            # Apply PCA
            pca_autoencoder = PCA(n_components=68)
            X_train = pca_autoencoder.fit_transform(X_train)
            X_val = pca_autoencoder.transform(X_val)

            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            X_val_tensor = torch.FloatTensor(X_val)

            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor)
            val_dataset = TensorDataset(X_val_tensor)

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            input_dim = X_train.shape[1]

            # Initialize the model
            autoencoder = BatchNormAutoencoder(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
                activation_type="LeakyReLU",
                output_activation_type="Sigmoid",
            ).to(device)

            # Set up optimizer and loss function
            optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
            criterion = nn.MSELoss()

            # Train the model
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

            print("Finished training Autoencoder")

            # Load best model
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path)
                autoencoder.load_state_dict(checkpoint["model_state_dict"])
                autoencoder.eval()

                # Apply PCA to full training data
                pca_ocsvm = PCA(n_components=68)
                X_train_full_pca = pca_ocsvm.fit_transform(X_train_full)

                # Get encoded representations of training data
                X_train_full_tensor = torch.FloatTensor(X_train_full_pca)
                X_train_full_dataset = TensorDataset(X_train_full_tensor)
                X_train_full_loader = DataLoader(
                    X_train_full_dataset, batch_size=batch_size, shuffle=False
                )

                X_train_encoded = []
                with torch.no_grad():
                    for data in X_train_full_loader:
                        data_x = data[0].to(device)
                        encoded = autoencoder.encode(data_x)
                        X_train_encoded.append(encoded.cpu().numpy())

                X_train_encoded = np.vstack(X_train_encoded)

                print("Training OCSVM")
                # Train OCSVM on encoded features
                ocsvm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.2)
                ocsvm.fit(X_train_encoded)
                print("Finished training OCSVM")

                # Transform and encode test data
                X_test_pca = pca_ocsvm.transform(X_test)
                X_test_tensor = torch.FloatTensor(X_test_pca).to(device)

                test_dataset = TensorDataset(X_test_tensor)
                X_test_loader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False
                )

                X_test_encoded = []
                with torch.no_grad():
                    for data in X_test_loader:
                        data_x = data[0].to(device)
                        encoded = autoencoder.encode(data_x)
                        X_test_encoded.append(encoded.cpu().numpy())

                X_test_encoded = np.vstack(X_test_encoded)

                # Make predictions and calculate accuracy
                print("Making predictions")
                y_pred = ocsvm.predict(X_test_encoded)
                accuracy = accuracy_score(y_test, y_pred)
                print("Finished making predictions")
                print(f"Accuracy: {accuracy:.4f}")

                # Check if this is the best accuracy for this configuration
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "accuracy": accuracy,
                        "hidden_dims": hidden_dims,
                        "latent_dim": latent_dim,
                        "model_path": best_model_path,
                    }
            else:
                print(f"Model training failed for this configuration")

    # Save the best result for this configuration
    if best_params is not None:
        best_results[config_name] = best_params

        # Log to file
        with open(log_file_path, "a") as f:
            f.write(f"{'-'*50}\n")
            f.write(f"Best result for {config_name}:\n")
            f.write(f"Accuracy: {best_params['accuracy']:.4f}\n")
            f.write(
                f"Model configuration: hidden_dims={best_params['hidden_dims']}, latent_dim={best_params['latent_dim']}\n"
            )
            f.write(
                f"Learning parameters: lr={best_params['learning_rate']}, batch_size={best_params['batch_size']}\n"
            )
            f.write(f"Model saved at: {best_params['model_path']}\n\n")
    else:
        print(f"No successful training for configuration {config_name}")

# Save all results to a JSON file for later analysis
with open(os.path.join(results_dir, f"all_results_{timestamp}.json"), "w") as f:
    json.dump(best_results, f, indent=4)

print(f"\nAll results have been saved to {log_file_path}")

import numpy as np
import pandas as pd
import torch
from models import BatchNormAutoencoder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import OneClassSVM
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
)
import optuna
import matplotlib.pyplot as plt
import random


def set_seed(seed, cudnn_deterministic=False):
    """Set all random seeds for reproducibility"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cudnn_deterministic:
        # Make cudnn deterministic (slightly lower performance but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def inner_objective(
    trial: optuna.Trial,
    dataset: pd.DataFrame,
    autoencoder: nn.Module,
) -> float:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = dataset.drop(
        ["attack_binary", "attack_categorical", "attack_class"], axis=1
    ).values
    y = dataset["attack_binary"].values

    # Extract features
    autoencoder.eval()
    X_tensor = torch.FloatTensor(X).to(device)

    # Extract in batches to prevent memory issues
    X_tensor_dataset = TensorDataset(X_tensor, torch.zeros(len(X_tensor)))
    normal_loader = DataLoader(X_tensor_dataset, batch_size=128)

    X_encoded = []
    with torch.no_grad():
        for data, _ in normal_loader:
            encoded = autoencoder.encode(data)
            X_encoded.append(encoded.cpu().numpy())

    X_encoded = np.vstack(X_encoded)

    nu = trial.suggest_float("nu", 0.01, 0.5)
    gamma = trial.suggest_float("gamma", 0.01, 1.0)

    kf = KFold(n_splits=7, shuffle=True, random_state=42)

    scores = []

    for train_idx, val_idx in kf.split(X_encoded):
        X_train, X_val = X_encoded[train_idx], X_encoded[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_normal = X_train[y_train == 1]

        ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)

        ocsvm.fit(X_train_normal)

        y_pred = ocsvm.predict(X_val)

        acc = accuracy_score(y_val, y_pred)

        scores.append(acc)

    return np.mean(scores)


def outer_objective(
    trial: optuna.Trial,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    train_dataset_unsplit: pd.DataFrame,
    input_dim: int,
    seed: int = 42,
    cudnn_deterministic: bool = False,
    verbose: bool = False,
    best_model_path="best_autoencoder_tuning.pth",
) -> float:
    """
    Outer objective function to optimize autoencoder hyperparameters
    """

    set_seed(seed, cudnn_deterministic)

    # Autoencoder hyperparameters to optimize
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 3)
    hidden_dims = []
    for i in range(num_hidden_layers):
        hidden_factor = trial.suggest_float(f"hidden_factor_{i}", 0.3, 0.8)
        prev_dim = input_dim if i == 0 else int(hidden_dims[-1])
        hidden_dims.append(int(prev_dim * hidden_factor))

    # Latent dimension as a fraction of the last hidden layer
    latent_factor = trial.suggest_float("latent_factor", 0.2, 0.8)
    latent_dim = max(2, int(hidden_dims[-1] * latent_factor))

    # Learning parameters
    lr = trial.suggest_categorical("lr", [0.1, 0.01, 0.001, 0.0001])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    # NOTE: not using this
    # activation_type = trial.suggest_categorical(
    #     "activation_type", ["ReLU", "LeakyReLU", "ELU"]
    # )
    # output_activation_type = trial.suggest_categorical(
    #     "output_activation_type", [None, "Sigmoid", "Tanh"]
    # )
    # weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    # optimizer_name = trial.suggest_categorical(
    #     "optimizer", ["Adam", "AdamW", "RMSprop"]
    # )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = BatchNormAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        activation_type="LeakyReLU",
        output_activation_type="Sigmoid",
    )

    # NOTE: not using this
    # Select optimizer
    # if optimizer_name == "Adam":
    #     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # elif optimizer_name == "AdamW":
    #     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # else:
    #     optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    # loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # best_model_path = "best_autoencoder_tuning.pth"

    # Train model
    # history = train_autoencoder(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     optimizer=optimizer,
    #     criterion=criterion,
    #     epochs=50,
    #     best_model_path=best_model_path,
    # )

    history, is_good_model = train_autoencoderV2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=50,
        best_model_path=best_model_path,
        verbose=True,
        early_stopping_patience=5,
        improvement_threshold=0.001,
        good_model_threshold=0.05,
    )

    if verbose:
        print("Finished training autoencoder")

    if not is_good_model:
        print("Skip OCSVM tuning due to poor performance of autoencoder")
        return 0.0

    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create inner Optuna study for OCSVM optimization
    inner_study = optuna.create_study(direction="maximize")
    inner_study.optimize(
        lambda inner_trial: inner_objective(
            inner_trial, dataset=train_dataset_unsplit, autoencoder=model
        ),
        n_trials=20,
    )

    # Get best OCSVM performance metric
    best_ocsvm_performance = inner_study.best_value
    best_ocsvm_params = inner_study.best_params

    # Store both autoencoder and OCSVM parameters for later reference
    trial.set_user_attr("best_ocsvm_params", best_ocsvm_params)
    trial.set_user_attr("final_val_loss", history["val_loss"][-1])
    trial.set_user_attr("latent_dim", latent_dim)
    trial.set_user_attr("hidden_dims", hidden_dims)

    # Return OCSVM performance as the objective to maximize
    return best_ocsvm_performance


def train_autoencoderV2(
    model,
    train_loader,
    optimizer,
    criterion,
    val_loader=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=10,
    best_model_path="./best_autoencoder_tuning.pth",
    verbose=False,
):
    model.to(device)

    if verbose:
        print("")
        print(f"Using device: {device}")
        print("Training autoencoder...")

    history = {"loss": [], "val_loss": []}
    model.train()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            batch_x = batch[0].to(device)

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history["loss"].append(avg_loss)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch[0].to(device)
                    val_outputs = model(val_inputs)
                    batch_val_loss = criterion(val_outputs, val_inputs).item()
                    val_loss += batch_val_loss

            avg_val_loss = val_loss / len(val_loader)
            history["val_loss"].append(avg_val_loss)

            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.10f}, Val Loss: {avg_val_loss:.10f}"
                )
            # Save model if validation loss improved
            if avg_val_loss < best_val_loss:
                improvement = best_val_loss - avg_val_loss

                # Save the model
                best_val_loss = avg_val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                    },
                    best_model_path,
                )

                if verbose:
                    print(f"✅ Model saved with val_loss: {best_val_loss:.10f}")

                model.train()
        else:
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.10f}")

            # If no validation set, save based on training loss
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": best_val_loss,
                    },
                    best_model_path,
                )

                if verbose:
                    print(f"✅ Model saved with train_loss: {best_val_loss:.10f}")

    plt.figure(figsize=(15, 10))
    plt.plot(history["loss"], label="Training Loss")
    if "val_loss" in history and history["val_loss"]:
        plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"Loss History (Epoch {epoch+1})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.close()

    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    return history


# this has early stopping
def train_autoencoderV3(
    model,
    train_loader,
    optimizer,
    criterion,
    val_loader=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=10,
    best_model_path="./best_autoencoder_tuning.pth",
    verbose=False,
    patience=5,  # Add patience parameter for early stopping
):
    model.to(device)

    if verbose:
        print("")
        print(f"Using device: {device}")
        print("Training autoencoder...")

    history = {"loss": [], "val_loss": []}
    model.train()

    best_val_loss = float("inf")
    patience_counter = 0  # Initialize patience counter

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            batch_x = batch[0].to(device)

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history["loss"].append(avg_loss)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch[0].to(device)
                    val_outputs = model(val_inputs)
                    batch_val_loss = criterion(val_outputs, val_inputs).item()
                    val_loss += batch_val_loss

            avg_val_loss = val_loss / len(val_loader)
            history["val_loss"].append(avg_val_loss)

            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.10f}, Val Loss: {avg_val_loss:.10f}"
                )
            # Save model if validation loss improved
            if avg_val_loss < best_val_loss:
                improvement = best_val_loss - avg_val_loss

                # Save the model
                best_val_loss = avg_val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                    },
                    best_model_path,
                )

                if verbose:
                    print(f"✅ Model saved with val_loss: {best_val_loss:.10f}")

                # Reset patience counter when improvement found
                patience_counter = 0

                model.train()
            else:
                # Increment patience counter if no improvement
                patience_counter += 1
                if verbose:
                    print(
                        f"No improvement in validation loss. Patience: {patience_counter}/{patience}"
                    )

                # Check if early stopping should be triggered
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                    break

                model.train()
        else:
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.10f}")

            # If no validation set, save based on training loss
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": best_val_loss,
                    },
                    best_model_path,
                )

                if verbose:
                    print(f"✅ Model saved with train_loss: {best_val_loss:.10f}")

                # Reset patience counter when improvement found
                patience_counter = 0
            else:
                # Increment patience counter if no improvement
                patience_counter += 1
                if verbose:
                    print(
                        f"No improvement in training loss. Patience: {patience_counter}/{patience}"
                    )

                # Check if early stopping should be triggered
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                    break

    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    return history


# def train_autoencoder(
#     model,
#     train_loader,
#     optimizer,
#     criterion,
#     val_loader=None,
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     epochs=10,
#     best_model_path="./best_autoencoder.pth",
#     verbose=False,
# ):
#     """
#     Train an autoencoder model and monitor performance.

#     Args:
#         autoencoder: The autoencoder model to train
#         train_loader: DataLoader for training data
#         val_loader: DataLoader for validation data (optional)
#         device: Device to train on ('cuda' or 'cpu')
#         epochs: Number of training epochs
#         learning_rate: Learning rate for optimizer
#         best_model_path: Path to save best model checkpoint

#     Returns:
#         tuple: (trained autoencoder, training history)
#     """

#     model.to(device)

#     if verbose:
#         print(f"Using device: {device}")
#         print("Training autoencoder...")

#     history = {"loss": [], "val_loss": []}
#     model.train()

#     best_val_loss = float("inf")
#     for epoch in range(epochs):
#         total_loss = 0.0
#         for batch in train_loader:
#             batch_x = batch[0].to(device)

#             # Forward pass
#             outputs = model(batch_x)
#             loss = criterion(outputs, batch_x)

#             # Backward pass and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_loss = total_loss / len(train_loader)
#         history["loss"].append(avg_loss)

#         if val_loader is not None:
#             model.eval()
#             val_loss = 0.0

#             with torch.no_grad():
#                 for val_batch in val_loader:
#                     val_inputs = val_batch[0].to(device)
#                     val_outputs = model(val_inputs)
#                     batch_val_loss = criterion(val_outputs, val_inputs).item()
#                     val_loss += batch_val_loss

#             avg_val_loss = val_loss / len(val_loader)
#             history["val_loss"].append(avg_val_loss)
#             if verbose:
#                 print(
#                     f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.10f}, Val Loss: {avg_val_loss:.10f}"
#                 )
#             # Save model if validation loss improved
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 torch.save(
#                     {
#                         "epoch": epoch,
#                         "model_state_dict": model.state_dict(),
#                         "optimizer_state_dict": optimizer.state_dict(),
#                         "val_loss": best_val_loss,
#                     },
#                     best_model_path,
#                 )
#                 if verbose:
#                     print(f"✅ Model saved with val_loss: {best_val_loss:.10f}")
#                 model.train()  # Switch back to training mode after validation
#         else:
#             # If no validation set, save based on training loss
#             if avg_loss < best_val_loss:
#                 best_val_loss = avg_loss
#                 torch.save(
#                     {
#                         "epoch": epoch,
#                         "model_state_dict": model.state_dict(),
#                         "optimizer_state_dict": optimizer.state_dict(),
#                         "train_loss": best_val_loss,
#                     },
#                     best_model_path,
#                 )
#                 if verbose:
#                     print(f"✅ Model saved with train_loss: {best_val_loss:.10f}")
#             if verbose:
#                 print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.10f}")

#         if epoch == epochs - 1:
#             plt.figure(figsize=(15, 10))

#             # Plot 1: Loss history
#             plt.subplot(2, 2, 1)
#             plt.plot(history["loss"], label="Training Loss")
#             if "val_loss" in history and history["val_loss"]:
#                 plt.plot(history["val_loss"], label="Validation Loss")
#             plt.title(f"Loss History (Epoch {epoch+1})")
#             plt.xlabel("Epoch")
#             plt.ylabel("Loss")
#             plt.legend()
#             plt.grid(True, alpha=0.3)
#             plt.show()

#             plt.close()

#     return history

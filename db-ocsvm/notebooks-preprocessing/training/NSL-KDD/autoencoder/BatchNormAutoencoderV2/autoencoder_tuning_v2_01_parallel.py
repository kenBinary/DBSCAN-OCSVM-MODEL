import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import optuna
from models import BatchNormAutoencoderV2
from tuning_utils import train_autoencoderV2, train_autoencoderV3
import os
from datetime import datetime
import multiprocessing
from functools import partial
import uuid

# Ensure output directory exists
os.makedirs("tuning_results", exist_ok=True)
os.makedirs("best_models", exist_ok=True)
results_file = (
    f"tuning_results/autoencoder_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_configurations = {
    0: {
        "hidden_dims": [64, 32],
        "latent_dim": 16,
    },
    1: {
        "hidden_dims": [96, 64, 48, 32],
        "latent_dim": 16,
    },
    2: {
        "hidden_dims": [64, 32],
        "latent_dim": 8,
    },
    3: {
        "hidden_dims": [100, 80, 60],
        "latent_dim": 20,
    },
    4: {
        "hidden_dims": [64, 32],
        "latent_dim": 12,
    },
    5: {
        "hidden_dims": [96, 64],
        "latent_dim": 55,
    },
    6: {
        "hidden_dims": [96, 72, 48, 32],
        "latent_dim": 16,
    },
    7: {
        "hidden_dims": [100, 64, 32],
        "latent_dim": 8,
    },
    8: {
        "hidden_dims": [116, 96, 64, 48],
        "latent_dim": 24,
    },
}

current_model_config = 0

hidden_dims_list = [model_configurations[current_model_config]["hidden_dims"]]
latent_dim_list = [model_configurations[current_model_config]["latent_dim"]]

test_run = True
if test_run:
    # Configuration parameters
    sample_size = 0.01
    use_sample = True
    epochs = 50
    ocsvm_trials = 10

    # # Training parameters
    # lr_list = [0.1, 0.01]
    # batch_size_list = [32, 64]
    # optimizer_list = ["adam"]
    # activation_type = ["ReLU"]
    # negative_slope = [0.01]
    # dropout_rate = [0.1]
    # weight_decay = [0]
    # Training parameters
    lr_list = [0.001, 0.0001]
    batch_size_list = [128]
    optimizer_list = ["adam", "adamw", "adadelta"]
    activation_type = ["LeakyReLU"]
    negative_slope = [0.2]
    dropout_rate = [0.1, 0.2]
    weight_decay = [0, 0.001, 0.000001]
else:
    # Configuration parameters
    sample_size = 0.01
    use_sample = False
    epochs = 50
    ocsvm_trials = 5

    # Training parameters
    lr_list = [0.1, 0.01, 0.001, 0.0001]
    batch_size_list = [32, 64, 128, 256]
    optimizer_list = ["adam", "rmsprop", "sgd", "adamw", "adagrad", "adadelta"]
    activation_type = ["ReLU", "LeakyReLU", "ELU", "GELU"]
    negative_slope = [0.01, 0.1, 0.2, 0.3, 0.4]
    dropout_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
    weight_decay = [0, 0.01, 0.001, 0.0001, 0.00001, 0.000001]


# Function to train and evaluate a single model configuration
def train_and_evaluate_model(
    params,
    X_train,
    X_train_full,
    X_val,
    X_test,
    y_test,
    device,
    epochs,
    ocsvm_trials,
    input_dim,
):
    hidden_dims = params["hidden_dims"]
    latent_dim = params["latent_dim"]
    optimizer_name = params["optimizer_name"]
    act_type = params["act_type"]
    ns = params["ns"]
    dr = params["dr"]
    wd = params["wd"]
    lr = params["lr"]
    batch_size = params["batch_size"]
    model_name = params["model_name"]

    model_id = str(uuid.uuid4())[:8]  # Generate unique ID for this run
    task_id = f"{model_id}_{model_name}_opt{optimizer_name}_lr{lr}_bs{batch_size}"
    print(
        f"\nTask {task_id}: Starting training with lr={lr}, batch_size={batch_size}, "
        f"optimizer={optimizer_name}, activation={act_type}"
    )

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)
    test_dataset = TensorDataset(X_test_tensor)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    X_test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Create model
    autoencoder = BatchNormAutoencoderV2(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        activation_type=act_type,
        negative_slope=ns,
        dropout_rate=dr,
        output_activation_type="Sigmoid",
    ).to(device)

    # Setup optimizer based on optimizer_name
    if optimizer_name == "adam":
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(autoencoder.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(autoencoder.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == "adagrad":
        optimizer = optim.Adagrad(autoencoder.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == "adadelta":
        optimizer = optim.Adadelta(autoencoder.parameters(), lr=lr, weight_decay=wd)

    criterion = nn.MSELoss()

    # Define model path with all parameters and unique ID
    best_model_path = f"best_models/autoencoder_{model_id}_hidden{hidden_dims}_latent{latent_dim}_opt{optimizer_name}_act{act_type}_ns{ns}_dr{dr}_wd{wd}_lr{lr}_bs{batch_size}.pth"

    print(f"Task {task_id}: Starting training autoencoder")
    # Train autoencoder
    history, is_good_model = train_autoencoderV3(
        model=autoencoder,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        val_loader=val_loader,
        epochs=epochs,
        best_model_path=best_model_path,
        verbose=False,
    )
    print(f"Task {task_id}: Training complete")

    # Load best model from training
    checkpoint = torch.load(best_model_path)
    autoencoder.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate model
    print(f"Task {task_id}: Training and evaluating OCSVM model with Optuna tuning")

    # Helper function to encode data with current model
    def encode_data(data_loader, autoencoder, device):
        encoded_data = []
        autoencoder.eval()
        with torch.no_grad():
            for data in data_loader:
                data_x = data[0].to(device)
                encoded = autoencoder.encode(data_x)
                encoded_data.append(encoded.cpu().numpy())
        return np.vstack(encoded_data)

    # Create data loader for training OCSVM
    X_train_full_tensor = torch.FloatTensor(X_train_full).to(device)
    train_full_dataset = TensorDataset(X_train_full_tensor)
    train_full_loader = DataLoader(train_full_dataset, batch_size=256, shuffle=False)

    # Encode data
    X_train_encoded = encode_data(train_full_loader, autoencoder, device)
    X_test_encoded = encode_data(X_test_loader, autoencoder, device)

    # Tune OCSVM with Optuna
    def objective(trial):
        nu = trial.suggest_float("nu", 0.01, 0.5)
        gamma = trial.suggest_float("gamma", 0.0001, 1.0)
        ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
        ocsvm.fit(X_train_encoded)
        y_pred = ocsvm.predict(X_test_encoded)
        acc = accuracy_score(y_test, y_pred)
        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=ocsvm_trials)

    # Get best parameters and train final model
    best_params = study.best_params
    best_ocsvm = OneClassSVM(
        kernel="rbf",
        nu=best_params.get("nu", 0.2),
        gamma=best_params.get("gamma", "auto"),
    )
    best_ocsvm.fit(X_train_encoded)

    # Make predictions and calculate metrics
    y_pred = best_ocsvm.predict(X_test_encoded)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=-1)
    recall = recall_score(y_test, y_pred, pos_label=-1)
    f1 = f1_score(y_test, y_pred, pos_label=-1)

    print(f"Task {task_id}: Finished with Accuracy: {acc:.4f}")

    # Return results
    return {
        "model_name": model_name,
        "hidden_dims": hidden_dims,  # Add hidden_dims to the result
        "latent_dim": latent_dim,  # Add latent_dim to the result
        "optimizer": optimizer_name,
        "activation": act_type,
        "negative_slope": ns if act_type == "LeakyReLU" else "N/A",
        "dropout_rate": dr,
        "weight_decay": wd,
        "lr": lr,
        "batch_size": batch_size,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ocsvm_params": best_params,
        "best_model_path": best_model_path,
    }


# Main function with multiprocessing
def run_grid_search():
    # Load and prepare data
    train_set_path = (
        "/home/jbct/Projects/thesis/db-ocsvm/data/processed/NSL-KDD/train_set_full.csv"
    )
    train_df = pd.read_csv(train_set_path)

    if use_sample:
        train_df = train_df.sample(frac=sample_size, random_state=42).reset_index(
            drop=True
        )

    print(f"Training data shape: {train_df.shape}")

    X_train_full = train_df.values

    # Split data into train and validation sets
    X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=42)
    X_train = X_train.values
    X_val = X_val.values

    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

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

    # Get input dimension
    input_dim = X_train.shape[1]

    # Initialize the log file with header
    with open(results_file, "w") as f:
        f.write("AUTOENCODER TUNING RESULTS\n")
        f.write("==========================\n\n")
        f.write("INDIVIDUAL TASK RESULTS\n")
        f.write("----------------------\n\n")

    # Track best results
    best_results = {}
    all_results = []

    # Calculate number of processes to use (leave one CPU core free)
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_processes} processes for parallel execution")

    # Make a list of all parameter combinations to test
    all_tasks = []
    for i, (hidden_dims, latent_dim) in enumerate(
        zip(hidden_dims_list, latent_dim_list)
    ):
        model_id = f"Model_{i+1}"
        model_name = f"{model_id}_hidden{hidden_dims}_latent{latent_dim}"

        for optimizer_name in optimizer_list:
            for act_type in activation_type:
                ns_values = negative_slope if act_type == "LeakyReLU" else [0.01]
                for ns in ns_values:
                    for dr in dropout_rate:
                        for wd in weight_decay:
                            for lr in lr_list:
                                for batch_size in batch_size_list:
                                    task = {
                                        "hidden_dims": hidden_dims,
                                        "latent_dim": latent_dim,
                                        "optimizer_name": optimizer_name,
                                        "act_type": act_type,
                                        "ns": ns,
                                        "dr": dr,
                                        "wd": wd,
                                        "lr": lr,
                                        "batch_size": batch_size,
                                        "model_name": model_name,
                                    }
                                    all_tasks.append(task)

    # Create partial function with fixed parameters
    worker_func = partial(
        train_and_evaluate_model,
        X_train=X_train,
        X_train_full=X_train_full,
        X_val=X_val,
        X_test=X_test,
        y_test=y_test,
        device=device,
        epochs=epochs,
        ocsvm_trials=ocsvm_trials,
        input_dim=input_dim,
    )

    # Run tasks in parallel
    total_tasks = len(all_tasks)
    print(f"Starting grid search with {total_tasks} parameter combinations")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(worker_func, all_tasks)):
            results.append(result)
            print(f"Completed {i+1}/{total_tasks} tasks")

            # Update best results for this model
            model_name = result["model_name"]
            if (
                model_name not in best_results
                or result["accuracy"] > best_results[model_name]["accuracy"]
            ):
                best_results[model_name] = result

            # Write this individual result to the log file immediately
            with open(results_file, "a") as f:
                f.write(f"Task {i+1}/{total_tasks} - {model_name}:\n")
                f.write(
                    f"  optimizer={result['optimizer']}, activation={result['activation']}, "
                )
                f.write(f"negative_slope={result['negative_slope']}, ")
                f.write(
                    f"dropout_rate={result['dropout_rate']}, weight_decay={result['weight_decay']}, "
                )
                f.write(f"lr={result['lr']}, batch_size={result['batch_size']}\n")
                f.write(
                    f"  accuracy={result['accuracy']:.8f}, precision={result['precision']:.8f}, "
                )
                f.write(f"recall={result['recall']:.8f}, f1={result['f1']:.8f}\n")
                f.write(f"  ocsvm_params={result['ocsvm_params']}\n")
                f.write(f"  model_path={result['best_model_path']}\n\n")

    # After all tasks are complete, write the summary results to the file
    with open(results_file, "a") as f:
        f.write("\n\nSUMMARY RESULTS\n")
        f.write("==============\n\n")

        # Group results by model configuration
        model_results = {}
        for result in results:
            model_name = result["model_name"]
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(result)

        # Write results for each model configuration
        for model_name, model_results_list in model_results.items():
            f.write(f"[{model_name}]\n")

            # Get the hidden_dims and latent_dim directly from the first result
            hidden_dims = model_results_list[0]["hidden_dims"]
            latent_dim = model_results_list[0]["latent_dim"]

            f.write(f"Hidden dimensions: {hidden_dims}\n")
            f.write(f"Latent dimension: {latent_dim}\n\n")
            f.write("Training parameters with good results:\n")

            # Log good results
            for result in model_results_list:
                if result["accuracy"] > 0.88 or test_run:
                    f.write(
                        f"  optimizer={result['optimizer']}, activation={result['activation']}, "
                        f"negative_slope={result['negative_slope']}, "
                        f"dropout_rate={result['dropout_rate']}, weight_decay={result['weight_decay']}, "
                        f"lr={result['lr']}, batch_size={result['batch_size']}: "
                        f"accuracy={result['accuracy']:.8f}, precision={result['precision']:.8f}, "
                        f"recall={result['recall']:.8f}, f1={result['f1']:.8f}, "
                        f"ocsvm_params={result['ocsvm_params']}\n"
                    )

            # Log best configuration for this model
            if model_name in best_results:
                best_result = best_results[model_name]
                f.write("\nBEST CONFIGURATION:\n")
                f.write("Best model saved to: " + best_result["best_model_path"] + "\n")
                f.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
                f.write(f"Precision: {best_result['precision']:.4f}\n")
                f.write(f"Recall: {best_result['recall']:.4f}\n")
                f.write(f"F1 Score: {best_result['f1']:.4f}\n")
                f.write(f"Optimizer: {best_result['optimizer']}\n")
                f.write(f"Activation: {best_result['activation']}\n")
                f.write(f"Negative Slope: {best_result['negative_slope']}\n")
                f.write(f"Dropout Rate: {best_result['dropout_rate']}\n")
                f.write(f"Weight Decay: {best_result['weight_decay']}\n")
                f.write(f"Learning rate: {best_result['lr']}\n")
                f.write(f"Batch size: {best_result['batch_size']}\n")
                f.write(f"Best OCSVM parameters: {best_result['ocsvm_params']}\n")

            f.write("\n" + "=" * 50 + "\n\n")

    # Print overall best results
    print("\nBEST RESULTS FOR EACH MODEL CONFIGURATION:")
    for model_name, config in best_results.items():
        print(
            f"{model_name}: accuracy={config['accuracy']:.4f}, "
            f"optimizer={config['optimizer']}, activation={config['activation']}, "
            f"negative_slope={config['negative_slope']}, dropout_rate={config['dropout_rate']}, "
            f"weight_decay={config['weight_decay']}, lr={config['lr']}, batch_size={config['batch_size']}"
        )

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    # This guard is necessary for multiprocessing to work properly
    run_grid_search()

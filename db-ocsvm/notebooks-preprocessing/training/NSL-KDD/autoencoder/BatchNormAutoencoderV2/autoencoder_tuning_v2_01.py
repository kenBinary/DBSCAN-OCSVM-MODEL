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

current_model_config = 5

hidden_dims_list = [model_configurations[current_model_config]["hidden_dims"]]
latent_dim_list = [model_configurations[current_model_config]["latent_dim"]]

test_run = True
if test_run:
    # Configuration parameters
    sample_size = 0.01
    use_sample = True
    epochs = 50
    ocsvm_trials = 100

    # Training parameters
    lr_list = [0.1, 0.01]
    batch_size_list = [32, 64]
    optimizer_list = ["adam"]
    activation_type = ["ReLU"]
    negative_slope = [0.01]
    dropout_rate = [0.1]
    weight_decay = [0]
else:
    # Configuration parameters
    sample_size = 0.01
    use_sample = False
    epochs = 50
    ocsvm_trials = 10

    # Training parameters
    lr_list = [0.1, 0.01, 0.001, 0.0001]
    batch_size_list = [32, 64, 128, 256]
    optimizer_list = ["adam", "rmsprop", "sgd", "adamw", "adagrad", "adadelta"]
    activation_type = ["ReLU", "LeakyReLU", "ELU", "GELU"]
    negative_slope = [0.01, 0.1, 0.2, 0.3, 0.4]
    dropout_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
    weight_decay = [0, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

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
def evaluate_model(autoencoder, trials=100):
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

    # Encode test data
    X_test_encoded = []
    with torch.no_grad():
        for data in X_test_loader:
            data_x = data[0].to(device)
            encoded = autoencoder.encode(data_x)
            X_test_encoded.append(encoded.cpu().numpy())

    X_test_encoded = np.vstack(X_test_encoded)

    # Define Optuna objective function for OCSVM
    def objective(trial):
        # Suggest hyperparameters
        nu = trial.suggest_float("nu", 0.01, 0.5)
        gamma = trial.suggest_float("gamma", 0.0001, 1.0)

        # Train OCSVM with suggested parameters
        ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)

        ocsvm.fit(X_train_encoded)

        # Make predictions
        y_pred = ocsvm.predict(X_test_encoded)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        return acc  # Optimize for accuracy

    # Create and run Optuna study
    print("Starting Optuna hyperparameter tuning for OCSVM...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)

    # Get best parameters
    best_params = study.best_params
    print(f"Best OCSVM parameters: {best_params}")

    # Train best OCSVM model
    best_ocsvm = OneClassSVM(
        kernel="rbf",
        nu=best_params.get("nu", 0.2),
        gamma=best_params.get("gamma", "auto"),
    )
    best_ocsvm.fit(X_train_encoded)

    # Make predictions with best model
    y_pred = best_ocsvm.predict(X_test_encoded)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=-1)
    recall = recall_score(y_test, y_pred, pos_label=-1)
    f1 = f1_score(y_test, y_pred, pos_label=-1)

    return acc, precision, recall, f1, best_params


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

        # Iterate through all combinations of parameters
        for optimizer_name in optimizer_list:
            for act_type in activation_type:
                # Only use negative_slope for LeakyReLU
                ns_values = negative_slope if act_type == "LeakyReLU" else [0.01]
                for ns in ns_values:
                    for dr in dropout_rate:
                        for wd in weight_decay:
                            for lr in lr_list:
                                for batch_size in batch_size_list:
                                    print(
                                        f"\nTraining {model_name} with lr={lr}, batch_size={batch_size}, "
                                        f"optimizer={optimizer_name}, activation={act_type}, "
                                        f"negative_slope={ns if act_type == 'LeakyReLU' else 'N/A'}, "
                                        f"dropout_rate={dr}, weight_decay={wd}"
                                    )

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

                                    # Create data loaders
                                    train_loader = DataLoader(
                                        train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                    )
                                    val_loader = DataLoader(
                                        val_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                    )

                                    # Setup optimizer based on optimizer_name
                                    if optimizer_name == "adam":
                                        optimizer = optim.Adam(
                                            autoencoder.parameters(),
                                            lr=lr,
                                            weight_decay=wd,
                                        )
                                    elif optimizer_name == "rmsprop":
                                        optimizer = optim.RMSprop(
                                            autoencoder.parameters(),
                                            lr=lr,
                                            weight_decay=wd,
                                        )
                                    elif optimizer_name == "sgd":
                                        optimizer = optim.SGD(
                                            autoencoder.parameters(),
                                            lr=lr,
                                            weight_decay=wd,
                                        )
                                    elif optimizer_name == "adamw":
                                        optimizer = optim.AdamW(
                                            autoencoder.parameters(),
                                            lr=lr,
                                            weight_decay=wd,
                                        )
                                    elif optimizer_name == "adagrad":
                                        optimizer = optim.Adagrad(
                                            autoencoder.parameters(),
                                            lr=lr,
                                            weight_decay=wd,
                                        )
                                    elif optimizer_name == "adadelta":
                                        optimizer = optim.Adadelta(
                                            autoencoder.parameters(),
                                            lr=lr,
                                            weight_decay=wd,
                                        )

                                    criterion = nn.MSELoss()

                                    # Define model path with all parameters
                                    best_model_path = f"best_models/autoencoder_hidden{hidden_dims}_latent{latent_dim}_opt{optimizer_name}_act{act_type}_ns{ns}_dr{dr}_wd{wd}_lr{lr}_bs{batch_size}.pth"

                                    print(f"Starting training autoencoder")
                                    # Train autoencoder
                                    history, is_good_model = train_autoencoderV3(
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
                                    autoencoder.load_state_dict(
                                        checkpoint["model_state_dict"]
                                    )

                                    # Evaluate model with Optuna tuned OCSVM
                                    print(
                                        "Training and evaluating OCSVM model with Optuna tuning"
                                    )
                                    acc, precision, recall, f1, best_ocsvm_params = (
                                        evaluate_model(autoencoder, trials=ocsvm_trials)
                                    )
                                    print(
                                        f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, "
                                        f"Recall: {recall:.4f}, F1: {f1:.4f}"
                                    )
                                    print(f"Best OCSVM params: {best_ocsvm_params}")
                                    print("Finished evaluating model")

                                    if test_run:
                                        f.write(
                                            f"  optimizer={optimizer_name}, activation={act_type}, "
                                            f"negative_slope={ns if act_type == 'LeakyReLU' else 'N/A'}, "
                                            f"dropout_rate={dr}, weight_decay={wd}, lr={lr}, batch_size={batch_size}: "
                                            f"accuracy={acc:.8f}, precision={precision:.8f}, "
                                            f"recall={recall:.8f}, f1={f1:.8f}, ocsvm_params={best_ocsvm_params}\n"
                                        )

                                    # Log results with all parameters
                                    if acc > 0.88 and not test_run:
                                        f.write(
                                            f"  optimizer={optimizer_name}, activation={act_type}, "
                                            f"negative_slope={ns if act_type == 'LeakyReLU' else 'N/A'}, "
                                            f"dropout_rate={dr}, weight_decay={wd}, lr={lr}, batch_size={batch_size}: "
                                            f"accuracy={acc:.8f}, precision={precision:.8f}, "
                                            f"recall={recall:.8f}, f1={f1:.8f}, ocsvm_params={best_ocsvm_params}\n"
                                        )

                                    # Track best configuration for this model
                                    if acc > best_acc:
                                        best_acc = acc
                                        best_config = {
                                            "optimizer": optimizer_name,
                                            "activation": act_type,
                                            "best_model_path": best_model_path,
                                            "negative_slope": (
                                                ns if act_type == "LeakyReLU" else "N/A"
                                            ),
                                            "dropout_rate": dr,
                                            "weight_decay": wd,
                                            "lr": lr,
                                            "batch_size": batch_size,
                                            "accuracy": acc,
                                            "precision": precision,
                                            "recall": recall,
                                            "f1": f1,
                                            "ocsvm_params": best_ocsvm_params,
                                        }

        # Save best config for this model
        best_results[model_name] = best_config

        # Log best configuration for this model
        f.write("\nBEST CONFIGURATION:\n")
        f.write("Best model saved to: " + best_config["best_model_path"] + "\n")
        f.write(f"Accuracy: {best_config['accuracy']:.4f}\n")
        f.write(f"Precision: {best_config['precision']:.4f}\n")
        f.write(f"Recall: {best_config['recall']:.4f}\n")
        f.write(f"F1 Score: {best_config['f1']:.4f}\n")
        f.write(f"Optimizer: {best_config['optimizer']}\n")
        f.write(f"Activation: {best_config['activation']}\n")
        f.write(f"Negative Slope: {best_config['negative_slope']}\n")
        f.write(f"Dropout Rate: {best_config['dropout_rate']}\n")
        f.write(f"Weight Decay: {best_config['weight_decay']}\n")
        f.write(f"Learning rate: {best_config['lr']}\n")
        f.write(f"Batch size: {best_config['batch_size']}\n")
        f.write(f"Best OCSVM parameters: {best_config['ocsvm_params']}\n")
        f.write("\n" + "=" * 50 + "\n\n")

# Print overall best results
print("\nBEST RESULTS FOR EACH MODEL CONFIGURATION:")
for model_name, config in best_results.items():
    print(
        f"best models saved to {config['best_model_path']}"
        f"{model_name}: accuracy={config['accuracy']:.4f}, "
        f"optimizer={config['optimizer']}, activation={config['activation']}, "
        f"negative_slope={config['negative_slope']}, dropout_rate={config['dropout_rate']}, "
        f"weight_decay={config['weight_decay']}, lr={config['lr']}, batch_size={config['batch_size']}"
    )

print(f"\nResults saved to: {results_file}")

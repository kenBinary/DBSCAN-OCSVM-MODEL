#############################################################
# Gives the best autoencoder configuration based on         #
#  reconstruction error of normal, anomaly, or difference   #
# Also exports the best models to ONNX format               #
#############################################################
import json
import torch
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from models import BatchNormAutoencoderV2
from tuning_utils import train_autoencoderV3
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

# Ensure output directory exists
os.makedirs("tuning_results", exist_ok=True)
results_file = (
    f"tuning_results/autoencoder_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model configurations
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
    validation_criterion = "difference"  # Options: "normal", "anomaly", "difference"

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
    validation_criterion = "difference"  # Options: "normal", "anomaly", "difference"

    # Training parameters
    lr_list = [0.01, 0.001]
    batch_size_list = [128, 256]
    optimizer_list = [
        "adam",
        "rmsprop",
        "adamw",
        "adadelta",
    ]
    activation_type = ["ReLU", "LeakyReLU", "ELU", "GELU"]
    negative_slope = [0.01, 0.2]
    dropout_rate = [0.1, 0.3]
    weight_decay = [0, 0.001]


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
X_train = X_train.values
X_val = X_val.values


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

        # Calculate total number of combinations for progress tracking
        slopes_count = len(negative_slope) if "LeakyReLU" in activation_type else 1
        total_combinations = (
            len(lr_list)
            * len(batch_size_list)
            * len(optimizer_list)
            * len(activation_type)
            * slopes_count
            * len(dropout_rate)
            * len(weight_decay)
        )
        current_combination = 0
        print(f"Total hyperparameter combinations to evaluate: {total_combinations}")

        saved_progress = {
            "finished_combinations": [],
            "best_config": None,
            "pytorch_path": None,
        }

        # Check if file exists before attempting to read it
        file_path = "tuning_results/saved_progress.json"
        if os.path.exists(file_path):
            # Reading from existing file
            with open(file_path, "r") as infile:
                saved_progress = json.load(infile)
            current_combination = len(saved_progress["finished_combinations"])
            best_config = saved_progress["best_config"]
            best_model_path = saved_progress.get("pytorch_path", None)

            if validation_criterion == "normal":
                best_metric = best_config[
                    "normal_loss"
                ]  # Lower is better for normal reconstruction
            elif validation_criterion == "anomaly":
                best_metric = best_config[
                    "anomaly_loss"
                ]  # Higher is better for anomaly reconstruction
            elif validation_criterion == "difference":
                best_metric = best_config[
                    "loss_difference"
                ]  # Higher difference is better
        else:
            # Create the file with an empty list if it doesn't exist
            with open(file_path, "w") as outfile:
                json.dump(saved_progress, outfile)

        # finished_combinations = []
        # # Check if file exists before attempting to read it
        # file_path = "tuning_results/finished_combinations.json"
        # if os.path.exists(file_path):
        #     # Reading from existing file
        #     with open(file_path, "r") as infile:
        #         finished_combinations = json.load(infile)
        #     current_combination = len(finished_combinations)
        # else:
        #     # Create the file with an empty list if it doesn't exist
        #     with open(file_path, "w") as outfile:
        #         json.dump(finished_combinations, outfile)

        # Iterate through all combinations of hyperparameters
        for lr in lr_list:
            for batch_size in batch_size_list:
                for optimizer_name in optimizer_list:
                    for act_type in activation_type:
                        # Only use negative_slope when activation is LeakyReLU
                        slopes_to_try = (
                            negative_slope if act_type == "LeakyReLU" else [None]
                        )
                        for slope in slopes_to_try:
                            for dropout in dropout_rate:
                                for decay in weight_decay:

                                    current_parameter_combination = f"{lr}_{batch_size}_{optimizer_name}_{act_type}_{slope}_{dropout}_{decay}"
                                    if (
                                        current_parameter_combination
                                        in saved_progress["finished_combinations"]
                                    ):
                                        print(
                                            f"Skipping finished combination: {current_parameter_combination}"
                                        )
                                        continue

                                    current_combination += 1
                                    print(
                                        f"\nCombination {current_combination}/{total_combinations} ({(current_combination/total_combinations)*100:.1f}%)"
                                    )
                                    print(f"Training {model_name} with:")
                                    print(f"  lr={lr}, batch_size={batch_size}")
                                    print(
                                        f"  optimizer={optimizer_name}, activation={act_type}"
                                    )
                                    if slope is not None:
                                        print(f"  negative_slope={slope}")
                                    print(
                                        f"  dropout_rate={dropout}, weight_decay={decay}"
                                    )

                                    # Create model with the current hyperparameters
                                    autoencoder = BatchNormAutoencoderV2(
                                        input_dim=input_dim,
                                        hidden_dims=hidden_dims,
                                        latent_dim=latent_dim,
                                        activation_type=act_type,
                                        negative_slope=slope,
                                        dropout_rate=dropout,
                                        output_activation_type="Sigmoid",
                                    ).to(device)

                                    # Create data loaders
                                    train_loader = DataLoader(
                                        train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=4,
                                        pin_memory=True,
                                        prefetch_factor=2,
                                    )
                                    val_loader = DataLoader(
                                        val_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                    )

                                    # Setup optimizer based on the selected optimizer type
                                    if optimizer_name == "adam":
                                        optimizer = optim.Adam(
                                            autoencoder.parameters(),
                                            lr=lr,
                                            weight_decay=decay,
                                        )
                                    elif optimizer_name == "rmsprop":
                                        optimizer = optim.RMSprop(
                                            autoencoder.parameters(),
                                            lr=lr,
                                            weight_decay=decay,
                                        )
                                    elif optimizer_name == "sgd":
                                        optimizer = optim.SGD(
                                            autoencoder.parameters(),
                                            lr=lr,
                                            weight_decay=decay,
                                        )
                                    elif optimizer_name == "adamw":
                                        optimizer = optim.AdamW(
                                            autoencoder.parameters(),
                                            lr=lr,
                                            weight_decay=decay,
                                        )
                                    elif optimizer_name == "adagrad":
                                        optimizer = optim.Adagrad(
                                            autoencoder.parameters(),
                                            lr=lr,
                                            weight_decay=decay,
                                        )
                                    elif optimizer_name == "adadelta":
                                        optimizer = optim.Adadelta(
                                            autoencoder.parameters(),
                                            lr=lr,
                                            weight_decay=decay,
                                        )

                                    criterion = nn.MSELoss()

                                    # Define model path for training with all hyperparameters
                                    param_str = f"{model_id}_hidden{hidden_dims}_latent{latent_dim}_lr{lr}_bs{batch_size}_opt{optimizer_name}_act{act_type}"
                                    if slope is not None:
                                        param_str += f"_slp{slope}"
                                    param_str += f"_dr{dropout}_wd{decay}"
                                    current_model_path = (
                                        f"best_models/autoencoder_{param_str}.pth"
                                    )
                                    os.makedirs("best_models", exist_ok=True)

                                    # Train autoencoder using the existing utility function
                                    print(f"Starting training autoencoder")
                                    history = train_autoencoderV3(
                                        model=autoencoder,
                                        train_loader=train_loader,
                                        val_loader=val_loader,
                                        optimizer=optimizer,
                                        criterion=criterion,
                                        epochs=epochs,
                                        best_model_path=current_model_path,
                                        verbose=True,
                                    )
                                    print(f"Training complete")

                                    # Load the best model from training
                                    checkpoint = torch.load(current_model_path)
                                    autoencoder.load_state_dict(
                                        checkpoint["model_state_dict"]
                                    )
                                    best_val_loss = checkpoint.get(
                                        "val_loss", float("inf")
                                    )

                                    # Evaluate model on test data
                                    print("Evaluating model on test set")
                                    eval_metrics = evaluate_model(autoencoder)
                                    normal_loss = eval_metrics["normal_loss"]
                                    anomaly_loss = eval_metrics["anomaly_loss"]
                                    loss_difference = eval_metrics["loss_difference"]

                                    print(
                                        f"Normal reconstruction loss: {normal_loss:.6f}"
                                    )
                                    print(
                                        f"Anomaly reconstruction loss: {anomaly_loss:.6f}"
                                    )
                                    print(
                                        f"Difference (anomaly - normal): {loss_difference:.6f}"
                                    )

                                    # Log results
                                    f.write(f"  {param_str}:\n")
                                    f.write(
                                        f"    Pytorch model saved in: {current_model_path} \n"
                                    )
                                    f.write(
                                        f"    Normal reconstruction loss: {normal_loss:.6f}\n"
                                    )
                                    f.write(
                                        f"    Anomaly reconstruction loss: {anomaly_loss:.6f}\n"
                                    )
                                    f.write(f"    Difference: {loss_difference:.6f}\n")
                                    f.write(f"    lr: {lr}, batch_size: {batch_size}\n")
                                    f.write(
                                        f"    optimizer: {optimizer_name}, activation{act_type}\n"
                                    )
                                    f.write(
                                        f"    dropout_rate: {dropout}, weight_decay: {decay}\n"
                                    )

                                    saved_progress["finished_combinations"].append(
                                        current_parameter_combination
                                    )
                                    # Writing to a file
                                    with open(
                                        "tuning_results/saved_progress.json", "w"
                                    ) as outfile:
                                        json.dump(saved_progress, outfile)

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
                                            "optimizer": optimizer_name,
                                            "activation": act_type,
                                            "negative_slope": slope,
                                            "dropout_rate": dropout,
                                            "weight_decay": decay,
                                            "val_loss": best_val_loss,
                                            "normal_loss": normal_loss,
                                            "anomaly_loss": anomaly_loss,
                                            "loss_difference": loss_difference,
                                        }
                                        best_model_path = current_model_path
                                        best_model_state = checkpoint[
                                            "model_state_dict"
                                        ]

                                        saved_progress["best_config"] = best_config
                                        saved_progress["pytorch_path"] = best_model_path
                                        with open(
                                            "tuning_results/saved_progress.json", "w"
                                        ) as outfile:
                                            json.dump(saved_progress, outfile)

        # Copy the best model for this configuration to a dedicated location and export to ONNX
        if best_model_path:
            # Save PyTorch model
            best_final_path = f"best_models/autoencoder_{model_id}.pth"
            os.makedirs("best_models", exist_ok=True)
            import shutil

            shutil.copy(best_model_path, best_final_path)
            print(f"Best model saved to {best_final_path}")

            # Add PyTorch model path to the configuration info
            best_config["pytorch_path"] = best_final_path

            # Create a fresh model instance with the best architecture and hyperparameters
            best_autoencoder = BatchNormAutoencoderV2(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                latent_dim=latent_dim,
                activation_type=best_config["activation"],
                negative_slope=best_config["negative_slope"],
                dropout_rate=best_config["dropout_rate"],
                output_activation_type="Sigmoid",
            ).to(device)

            # Load the state dict
            checkpoint = torch.load(best_model_path)
            best_autoencoder.load_state_dict(checkpoint["model_state_dict"])

            # Export to ONNX
            onnx_path = f"onnx_models/autoencoder_{model_id}.onnx"
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
        f.write(f"Optimizer: {best_config['optimizer']}\n")
        f.write(f"Activation: {best_config['activation']}\n")
        if best_config["negative_slope"] is not None:
            f.write(f"Negative slope: {best_config['negative_slope']}\n")
        f.write(f"Dropout rate: {best_config['dropout_rate']}\n")
        f.write(f"Weight decay: {best_config['weight_decay']}\n")
        f.write(f"PyTorch model path: {best_config.get('pytorch_path', 'Not saved')}\n")
        f.write(f"ONNX model path: {best_config.get('onnx_path', 'Not exported')}\n")
        f.write("\n" + "=" * 50 + "\n\n")

# Print overall best results
print(
    f"\nBEST RESULTS FOR EACH MODEL CONFIGURATION (criterion: {validation_criterion}):"
)
for model_name, config in best_results.items():
    print(f"{model_name}:")
    print(
        f"  {validation_criterion}_metric: {config.get(f'{validation_criterion}_loss' if validation_criterion != 'difference' else 'loss_difference'):.6f}"
    )
    print(f"  lr={config['lr']}, batch_size={config['batch_size']}")
    print(f"  optimizer={config['optimizer']}, activation={config['activation']}")
    if config["negative_slope"] is not None:
        print(f"  negative_slope={config['negative_slope']}")
    print(
        f"  dropout_rate={config['dropout_rate']}, weight_decay={config['weight_decay']}"
    )
    print(f"  PyTorch: {config.get('pytorch_path', 'Not saved')}")
    print(f"  ONNX: {config.get('onnx_path', 'Not exported')}")
    print("")

print(f"\nResults saved to: {results_file}")
print("ONNX models exported to 'onnx_models/' directory")

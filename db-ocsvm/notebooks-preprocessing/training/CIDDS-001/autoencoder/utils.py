import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import onnx
import onnxruntime
from tqdm import tqdm


def train_autoencoder(
    model,
    dataloader,
    val_loader=None,  # Now accepts validation dataloader directly
    epochs=30,
    learning_rate=0.0001,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_plots=True,
):

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    reconstruction_criterion = nn.MSELoss()

    history = {"loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Add tqdm progress bar for training
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=True
        )

        for data in progress_bar:
            inputs = data[0].to(device)

            # Forward pass
            outputs, encoded = model(inputs)

            # Calculate loss
            reconstruction_loss = reconstruction_criterion(outputs, inputs)
            sparsity_loss = model.kl_divergence_loss(encoded)
            total_loss = reconstruction_loss + model.sparsity_weight * sparsity_loss

            # Backward pass and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

            # Update progress bar with current loss
            progress_bar.set_postfix({"batch_loss": total_loss.item()})

        avg_train_loss = train_loss / len(dataloader)
        history["loss"].append(avg_train_loss)

        # Calculate validation loss if validation dataloader is provided
        if val_loader is not None:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch[0].to(device)
                    val_outputs, _ = model(val_inputs)
                    batch_val_loss = reconstruction_criterion(
                        val_outputs, val_inputs
                    ).item()
                    val_loss += batch_val_loss

            avg_val_loss = val_loss / len(val_loader)
            history["val_loss"].append(avg_val_loss)
            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )
            print("")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}")
            print("")

        # Plot every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            plt.figure(figsize=(15, 10))

            # Plot 1: Loss history
            plt.subplot(2, 2, 1)
            plt.plot(history["loss"], label="Training Loss")
            if "val_loss" in history and history["val_loss"]:
                plt.plot(history["val_loss"], label="Validation Loss")
            plt.title(f"Loss History (Epoch {epoch+1})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot 2: Sample reconstructions
            plt.subplot(2, 2, 2)
            model.eval()
            with torch.no_grad():
                # Get a batch of data
                sample_batch = next(iter(dataloader))[0][:5].to(
                    device
                )  # Take 5 samples
                reconstructed, _ = model(sample_batch)

                # Reshape data for visualization if needed
                # For NSL-KDD, it's usually just feature vectors, so we'll visualize as heatmaps
                sample_data = sample_batch.cpu().numpy()
                recon_data = reconstructed.cpu().numpy()

                # Plot sample and reconstruction side by side
                plt.imshow(
                    np.vstack((sample_data, recon_data)), aspect="auto", cmap="viridis"
                )
                plt.axhline(y=sample_data.shape[0] - 0.5, color="r", linestyle="-")
                plt.title("Original (top) vs Reconstructed (bottom)")
                plt.colorbar(label="Feature Value")

            # Plot 3: Bottleneck activations
            plt.subplot(2, 2, 3)
            with torch.no_grad():
                # Get bottleneck activations
                _, bottleneck = model(sample_batch)
                bottleneck = bottleneck.cpu().numpy()

                plt.imshow(bottleneck, aspect="auto", cmap="plasma")
                plt.title("Bottleneck Activations")
                plt.colorbar(label="Activation Value")
                plt.xlabel("Bottleneck Feature")
                plt.ylabel("Sample")

            # Plot 4: Reconstruction error per feature
            plt.subplot(2, 2, 4)
            with torch.no_grad():
                recon_error = (
                    torch.mean((sample_batch - reconstructed) ** 2, dim=0).cpu().numpy()
                )
                plt.bar(range(len(recon_error)), recon_error)
                plt.title("Avg Reconstruction Error per Feature")
                plt.xlabel("Feature Index")
                plt.ylabel("MSE")

            plt.tight_layout()

            if save_plots:
                plt.savefig(f"autoencoder_epoch_{epoch+1}.png")
                print(f"Plot saved for epoch {epoch+1}")
                print("")

            plt.close()

    return history


# def train_autoencoder(
#     model,
#     dataloader,
#     val_data=None,  # Optional validation data (X_val_tensor) contains ONLY normal data
#     epochs=30,
#     learning_rate=0.0001,
#     device="cuda" if torch.cuda.is_available() else "cpu",
# ):
#     model = model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     reconstruction_criterion = nn.MSELoss()

#     history = {"loss": [], "val_loss": []}

#     for epoch in range(epochs):
#         model.train()
#         train_loss = 0.0

#         for data in dataloader:
#             inputs = data[0].to(device)

#             # Forward pass
#             outputs, encoded = model(inputs)

#             # Calculate loss
#             reconstruction_loss = reconstruction_criterion(outputs, inputs)
#             sparsity_loss = model.kl_divergence_loss(encoded)
#             total_loss = reconstruction_loss + model.sparsity_weight * sparsity_loss

#             # Backward pass and optimize
#             optimizer.zero_grad()
#             total_loss.backward()
#             optimizer.step()

#             train_loss += total_loss.item()

#         avg_train_loss = train_loss / len(dataloader)
#         history["loss"].append(avg_train_loss)

#         # Calculate validation loss if validation data is provided
#         if val_data is not None:
#             model.eval()
#             with torch.no_grad():
#                 val_outputs, _ = model(val_data.to(device))
#                 val_loss = reconstruction_criterion(
#                     val_outputs, val_data.to(device)
#                 ).item()
#                 history["val_loss"].append(val_loss)
#                 print(
#                     f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
#                 )
#         else:
#             print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}")

#         # Plot every 5 epochs
#         if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
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

#             # Plot 2: Sample reconstructions
#             plt.subplot(2, 2, 2)
#             model.eval()
#             with torch.no_grad():
#                 # Get a batch of data
#                 sample_batch = next(iter(dataloader))[0][:5].to(
#                     device
#                 )  # Take 5 samples
#                 reconstructed, _ = model(sample_batch)

#                 # Reshape data for visualization if needed
#                 # For NSL-KDD, it's usually just feature vectors, so we'll visualize as heatmaps
#                 sample_data = sample_batch.cpu().numpy()
#                 recon_data = reconstructed.cpu().numpy()

#                 # Plot sample and reconstruction side by side
#                 plt.imshow(
#                     np.vstack((sample_data, recon_data)), aspect="auto", cmap="viridis"
#                 )
#                 plt.axhline(y=sample_data.shape[0] - 0.5, color="r", linestyle="-")
#                 plt.title("Original (top) vs Reconstructed (bottom)")
#                 plt.colorbar(label="Feature Value")

#             # Plot 3: Bottleneck activations
#             plt.subplot(2, 2, 3)
#             with torch.no_grad():
#                 # Get bottleneck activations
#                 _, bottleneck = model(sample_batch)
#                 bottleneck = bottleneck.cpu().numpy()

#                 plt.imshow(bottleneck, aspect="auto", cmap="plasma")
#                 plt.title("Bottleneck Activations")
#                 plt.colorbar(label="Activation Value")
#                 plt.xlabel("Bottleneck Feature")
#                 plt.ylabel("Sample")

#             # Plot 4: Reconstruction error per feature
#             plt.subplot(2, 2, 4)
#             with torch.no_grad():
#                 recon_error = (
#                     torch.mean((sample_batch - reconstructed) ** 2, dim=0).cpu().numpy()
#                 )
#                 plt.bar(range(len(recon_error)), recon_error)
#                 plt.title("Avg Reconstruction Error per Feature")
#                 plt.xlabel("Feature Index")
#                 plt.ylabel("MSE")

#             plt.tight_layout()
#             plt.savefig(f"autoencoder_epoch_{epoch+1}.png")
#             plt.close()

#             print(f"Plot saved for epoch {epoch+1}")

#     return history


def evaluate_reconstruction_error(
    model,
    X_test,
    y_test,
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_type="dsd_autoencoder",
):
    """
    Evaluate reconstruction error on test data, separating normal and anomalous samples.

    Parameters:
    -----------
    model : DeepSparseDenoisingAutoencoder
        The trained autoencoder model
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test labels (1 for normal, -1 for anomaly)
    device : torch.device
        Device to run the model on
    model_type: str
        Type of autoencoder model used ("dsd_autoencoder" or "batch_norm_autoencoder")

    Returns:
    --------
    dict
        Dictionary containing reconstruction errors and performance metrics
    """
    model.eval()

    # Convert test data to tensors
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    # Get indices for normal and anomalous samples
    normal_idx = y_test == 1
    anomaly_idx = y_test == -1

    # Extract normal and anomalous samples
    X_test_normal = X_test_tensor[normal_idx]
    X_test_anomaly = X_test_tensor[anomaly_idx]

    results = {}

    with torch.no_grad():
        # Process all test data
        if model_type == "dsd_autoencoder":
            reconstructed_all, bottleneck_all = model(X_test_tensor)
        elif model_type == "batch_norm_autoencoder":
            reconstructed_all = model(X_test_tensor)
            bottleneck_all = model.encode(X_test_tensor)

        mse_all = (
            torch.mean(torch.pow(X_test_tensor - reconstructed_all, 2), dim=1)
            .cpu()
            .numpy()
        )
        results["reconstruction_error_all"] = mse_all
        results["bottleneck_features"] = bottleneck_all.cpu().numpy()

        # Process normal samples
        if len(X_test_normal) > 0:
            if model_type == "dsd_autoencoder":
                reconstructed_normal, _ = model(X_test_normal)
            elif model_type == "batch_norm_autoencoder":
                reconstructed_normal = model(X_test_normal)

            mse_normal = (
                torch.mean(torch.pow(X_test_normal - reconstructed_normal, 2), dim=1)
                .cpu()
                .numpy()
            )
            results["reconstruction_error_normal"] = mse_normal
            results["avg_error_normal"] = np.mean(mse_normal)

        # Process anomalous samples
        if len(X_test_anomaly) > 0:
            if model_type == "dsd_autoencoder":
                reconstructed_anomaly, _ = model(X_test_anomaly)
            elif model_type == "batch_norm_autoencoder":
                reconstructed_anomaly = model(X_test_anomaly)

            mse_anomaly = (
                torch.mean(torch.pow(X_test_anomaly - reconstructed_anomaly, 2), dim=1)
                .cpu()
                .numpy()
            )
            results["reconstruction_error_anomaly"] = mse_anomaly
            results["avg_error_anomaly"] = np.mean(mse_anomaly)

    # Print summary statistics
    if "avg_error_normal" in results and "avg_error_anomaly" in results:
        print(f"Avg Reconstruction Error (Normal): {results['avg_error_normal']:.4f}")
        print(f"Avg Reconstruction Error (Anomaly): {results['avg_error_anomaly']:.4f}")
        print(
            f"Error Ratio (Anomaly/Normal): {results['avg_error_anomaly']/results['avg_error_normal']:.2f}x"
        )

    # You could also visualize the distribution of reconstruction errors
    return results


def extract_features(model, data_loader, device):
    features = []

    with torch.no_grad():
        for batch_x in data_loader:
            batch_x = batch_x[0].to(device)
            encoded = model.get_bottleneck_representation(batch_x)
            features.append(encoded.cpu().numpy())

    return np.vstack(features)


def export_autoencoder_to_onnx(model, input_size):
    """
    Export the full autoencoder model to ONNX format.

    Parameters:
    -----------
    model : DeepSparseDenoisingAutoencoder
        The trained autoencoder model
    input_size : int
        Number of input features
    file_path : str
        Output file path for the ONNX model
    """
    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor of the correct shape
    dummy_input = torch.randn(1, input_size)
    output_path = "/home/jbct/Projects/thesis/db-ocsvm/models/autoencoder/deep_sparse_denoising.onnx"

    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        output_path,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=True,  # optimization: fold constant values
        input_names=["input"],  # the model's input names
        output_names=["output", "latent"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
            "latent": {0: "batch_size"},
        },
    )

    print(f"Model exported to {output_path}")

    # Verify the export
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("Model verified - ONNX export successful!")


def extract_latent_representation_onnx(data, onnx_path="encoder.onnx"):
    """
    Extract latent representation using the ONNX encoder model.

    Parameters:
    -----------
    data : numpy.ndarray
        Input data as a numpy array
    onnx_path : str
        Path to the ONNX encoder model

    Returns:
    --------
    numpy.ndarray
        Latent representation (bottleneck features)
    """
    # Create an ONNX Runtime session
    session = onnxruntime.InferenceSession(onnx_path)

    # Get the input name
    input_name = session.get_inputs()[0].name

    # Run inference
    latent = session.run(None, {input_name: data.astype(np.float32)})[0]

    return latent


def run_autoencoder_onnx(data, onnx_path="autoencoder.onnx"):
    """
    Run full autoencoder inference using the ONNX model.

    Parameters:
    -----------
    data : numpy.ndarray
        Input data as a numpy array
    onnx_path : str
        Path to the ONNX autoencoder model

    Returns:
    --------
    tuple
        (reconstructed_output, latent_representation)
    """
    # Create an ONNX Runtime session
    session = onnxruntime.InferenceSession(onnx_path)

    # Get the input name
    input_name = session.get_inputs()[0].name

    # Get the output names
    output_names = [output.name for output in session.get_outputs()]

    # Run inference
    outputs = session.run(output_names, {input_name: data.astype(np.float32)})

    # Unpack the outputs - first is reconstructed, second is latent
    reconstructed = outputs[0]
    latent = outputs[1]

    return reconstructed, latent


def extract_latent_representation_batched(data, batch_size=1000):
    """
    Extract latent representations in batches for large datasets.

    Parameters:
    -----------
    data : numpy.ndarray
        Input data
    batch_size : int
        Batch size for processing
    onnx_path : str
        Path to the ONNX encoder model

    Returns:
    --------
    numpy.ndarray
        Latent representations for all data
    """
    output_path = "/home/jbct/Projects/thesis/db-ocsvm/models/autoencoder/deep_sparse_denoising.onnx"

    session = onnxruntime.InferenceSession(output_path)
    input_name = session.get_inputs()[0].name

    n_samples = data.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division

    latent_features = []

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)

        batch_data = data[start_idx:end_idx].astype(np.float32)
        batch_latent = session.run(None, {input_name: batch_data})[0]

        latent_features.append(batch_latent)

        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            print(f"Processed {end_idx}/{n_samples} samples")

    # Concatenate all batches
    return np.vstack(latent_features)

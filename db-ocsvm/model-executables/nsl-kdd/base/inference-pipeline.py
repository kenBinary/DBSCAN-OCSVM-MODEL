import pandas as pd
import onnxruntime as ort
import numpy as np
import pprint
import random
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run OCSVM inference pipeline for NSL-KDD."
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="Enable debug output"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./dataset.csv",
        help="Path to test set CSV",
    )
    parser.add_argument(
        "--ocsvm",
        type=str,
        default="./ocsvm.onnx",
        help="Path to OCSVM ONNX model",
    )
    args = parser.parse_args()

    debug = args.debug
    test_set_path = args.dataset
    ocsvm_onnx_path = args.ocsvm

    test_df = pd.read_csv(test_set_path)

    if debug:
        random_num = random.random()
        test_df = test_df.sample(frac=random_num, random_state=42).reset_index(
            drop=True
        )

    # Splitting into X and y
    X_test = test_df.drop(
        columns=["attack_binary", "attack_categorical", "attack_class"]
    ).values
    y_test = test_df["attack_binary"].values
    y_test_class = test_df["attack_class"].values

    test_set_count = test_df.shape[0]
    test_set_features = test_df.shape[1] - 3  # Exclude the target columns
    attack_classes = test_df["attack_class"].value_counts()
    attack_classes_dict = attack_classes.to_dict()

    if debug:
        print(f"test set count: {test_set_count} with {test_set_features} features")
        print(f"unique values: {attack_classes}")
        print(attack_classes_dict)
        print("")

    # Create an ONNX Runtime session
    session = ort.InferenceSession(ocsvm_onnx_path)

    # Prepare some test data
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    y_pred = session.run([output_name], {input_name: X_test.astype(np.float32)})[0]

    # Calculate detection rates for each class
    if debug:
        print("Detection rates by class:")

    class_metrics = {}
    for cls in np.unique(y_test_class):
        # Get indices for this class
        class_indices = y_test_class == cls

        # True values and predictions for this class
        y_true_cls = y_test[class_indices]
        y_pred_cls = y_pred[class_indices]

        # Calculate metrics
        if cls == "normal":
            # For normal class, we want to detect 1 (normal)
            correct = np.sum((y_pred_cls == -1))
            precision = precision_score(
                y_true_cls, y_pred_cls, pos_label=1, zero_division=0
            )
            recall = recall_score(y_true_cls, y_pred_cls, pos_label=1, zero_division=0)
        else:
            # For attack classes, we want to detect -1 (anomaly)
            correct = np.sum((y_pred_cls == -1))
            precision = precision_score(
                y_true_cls, y_pred_cls, pos_label=-1, zero_division=0
            )
            recall = recall_score(y_true_cls, y_pred_cls, pos_label=-1, zero_division=0)

        total = len(y_pred_cls)
        detection_rate = correct / total
        f1 = f1_score(
            y_true_cls,
            y_pred_cls,
            pos_label=-1 if cls != "normal" else 1,
            zero_division=0,
        )

        class_metrics[cls] = {
            "detection_rate": f"{detection_rate * 100:.2f}",
            "count": total,
            "correctly_detected": int(correct),
        }
        if debug:
            print(f"{cls}: {detection_rate * 100 :.2f} ({correct}/{total})")

    if debug:
        print("")
        pprint.pprint(class_metrics)
        print("")

    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
    if debug:
        print("Confusion Matrix:")
        print(cm)
        print("")

    cm_dict = {
        "True Anomaly": int(cm[0, 0]),
        "False Anomaly": int(cm[0, 1]),
        "True Normal": int(cm[1, 1]),
        "False Normal": int(cm[1, 0]),
    }
    if debug:
        pprint.pprint(cm_dict)
        print("")

    if debug:
        print("Classification Report ONNX:")
        print(classification_report(y_test, y_pred, target_names=["Anomaly", "Normal"]))
        print("")

    precision = precision_score(y_test, y_pred, pos_label=-1) * 100
    recall = recall_score(y_test, y_pred, pos_label=-1) * 100
    f1 = f1_score(y_test, y_pred, pos_label=-1) * 100
    accuracy = accuracy_score(y_test, y_pred) * 100

    if debug:
        print(f"Precision: {precision:.2f}%")
        print(f"Recall: {recall:.2f}%")
        print(f"F1 Score: {f1:.2f}%")
        print(f"Accuracy: {accuracy:.2f}%")

    # Create a structured results dictionary similar to CIDDS-001 version
    results = {
        "test_set": {
            "count": test_set_count,
            "features": test_set_features,
            "class_distribution": attack_classes_dict,
        },
        "detection_rates": class_metrics,
        "confusion_matrix": cm_dict,
        "metrics": {
            "precision": f"{precision:.2f}",
            "recall": f"{recall:.2f}",
            "f1_score": f"{f1:.2f}",
            "accuracy": f"{accuracy:.2f}",
        },
    }

    if not debug:
        pprint.pprint(results)

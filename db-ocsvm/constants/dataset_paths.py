import os
from typing import TypedDict, Literal, Union


class DatasetFiles(TypedDict):
    train: str
    test_unsplit: str
    test: str
    validation: str


class DatasetFilesRawNSL(TypedDict):
    train: str
    test: str


class DatasetFilesRawCIDDS(TypedDict):
    openstack: str
    external: str


class DatasetStructure(TypedDict):
    processed: dict[Literal["NSL-KDD", "CIDDS-001"], DatasetFiles]
    raw: dict[
        Literal["NSL-KDD", "CIDDS-001"], Union[DatasetFilesRawNSL, DatasetFilesRawCIDDS]
    ]
    sample: dict[Literal["NSL-KDD"], DatasetFiles]


def get_dataset_path(BASE_DIR, containing_folder, dataset_name):
    return os.path.join(BASE_DIR, containing_folder, dataset_name)


# how to get rooth path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_DIR = os.path.join(DATA_DIR, "raw")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")


DATASET: DatasetStructure = {
    "processed": {
        "NSL-KDD": {
            "train": get_dataset_path(PROCESSED_DIR, "NSL-KDD", "train_set.csv"),
            "test_unsplit": get_dataset_path(
                PROCESSED_DIR, "NSL-KDD", "test_set_unsplit.csv"
            ),
            "test": get_dataset_path(PROCESSED_DIR, "NSL-KDD", "test_set.csv"),
            "validation": get_dataset_path(
                PROCESSED_DIR, "NSL-KDD", "validation_set.csv"
            ),
        },
        "CIDDS-001": {
            "train": get_dataset_path(PROCESSED_DIR, "CIDDS-001", "train_set.csv"),
            "test_unsplit": get_dataset_path(
                PROCESSED_DIR, "CIDDS-001", "test_set_unsplit.csv"
            ),
            "test": get_dataset_path(PROCESSED_DIR, "CIDDS-001", "test_set.csv"),
            "validation": get_dataset_path(
                PROCESSED_DIR, "CIDDS-001", "validation_set.csv"
            ),
        },
    },
    "raw": {
        "NSL-KDD": {
            "train": get_dataset_path(RAW_DIR, "NSL-KDD", "KDDTrain+.csv"),
            "test": get_dataset_path(RAW_DIR, "NSL-KDD", "KDDTest+.csv"),
        },
        "CIDDS-001": {
            "openstack": get_dataset_path(
                RAW_DIR, "CIDDS-001", "cidds-001-openstack.parquet"
            ),
            "external": get_dataset_path(
                RAW_DIR, "CIDDS-001", "cidds-001-externalserver.parquet"
            ),
        },
    },
}

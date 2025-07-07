import os


def delete_file_os(file_path):
    """
    Delete a file using the os module.

    :param file_path: Path to the file to be deleted.
    """
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} has been deleted.")
    else:
        print(f"{file_path} does not exist.")

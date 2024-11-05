import os
import shutil
import subprocess
from typing import List

import nbformat

ASSIGNMENTS_FOLDER = "src/assignments"
NOTEBOOK_FILES = [
    "knn_part_1.ipynb",
    "knn_part_2.ipynb",
    "linear_part_1.ipynb",
    "linear_part_2.ipynb",
    "mlp_part_1.ipynb",
    "mlp_part_2.ipynb",
]

scripts_dir = os.path.join(os.path.dirname(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(scripts_dir, os.pardir, os.pardir))

ZIP_FILE = "hw1.zip"


def preprocess_notebooks(
    input_files: List[str], output_files: List[str], root_path: str
) -> None:
    """This function preprocesses the Jupyter notebooks by replacing the relative paths of the images
    with absolute paths.

    Example:
        <img src="relative/path.png" ...> -> <img src="root_path/relative/path.png" ...>

    Args:
        input_files (List[str]): List of input Jupyter notebook files.
        output_files (List[str]): List of output Jupyter notebook files.
        root_path (str): Path to the root folder.

    Returns:
        None
    """

    for input_path, output_path in zip(input_files, output_files):
        with open(input_path, "r") as infile:
            notebook = nbformat.read(infile, as_version=4)

        for cell in notebook.cells:
            if cell.cell_type == "markdown":
                cell.source = cell.source.replace(
                    '<img src="', f'<img src="{root_path}/'
                )

            with open(output_path, "w") as outfile:
                nbformat.write(notebook, outfile)

        print(f"\tINFO: Preprocessed {os.path.basename(input_path)}")


def create_html_files(jupyter_files: list, remove_original: bool = True):
    """This function creates HTML files from Jupyter notebooks.

    Args:
        jupyter_files (list): List of Jupyter notebook files.
        remove_original (bool, optional): Whether to remove the original Jupyter notebook files. Defaults to True.

    Returns:
        List[str]: List of HTML files.

    """

    for f in jupyter_files:
        subprocess.run(["jupyter", "nbconvert", "--to", "html", f])
        print(f"\tINFO: Created {os.path.basename(f).replace('.ipynb', '.html')}")

    if remove_original:
        for f in jupyter_files:
            os.remove(f)


def submit():
    output_file = os.path.join(PROJECT_DIR, ZIP_FILE)

    notebooks_folder = os.path.join(PROJECT_DIR, "notebooks")
    assignments_folder = os.path.join(PROJECT_DIR, "src", "assignments")

    # html_files = [os.path.join(root_path, f) for f in HTML_FILES]
    notebook_files = [os.path.join(PROJECT_DIR, f) for f in NOTEBOOK_FILES]
    preprocessed_notebook_files = [
        os.path.join(notebooks_folder, f) for f in NOTEBOOK_FILES
    ]

    # Change the .ipynb files to .html files


    # Create the notebooks folder
    print(f"\n================= SETUP =================\n")
    if not os.path.isdir(notebooks_folder):
        os.makedirs(notebooks_folder, exist_ok=True)
        print(f"\tINFO: Created {notebooks_folder}.")
    else:
        print(f"\tERROR: {notebooks_folder} already exists. Exiting...")
        exit(1)

    # Preprocess the notebooks
    print(f"\n================= PREPROCESSING =================\n")
    preprocess_notebooks(notebook_files, preprocessed_notebook_files, PROJECT_DIR)

    # Create the HTML files
    print(f"\n================= CREATING HTML FILES =================\n")
    create_html_files(preprocessed_notebook_files, remove_original=False)

    # List the files in the preprocessed notebooks folder
    print(f"\n================= LISTING FILES =================\n")
    subprocess.run(["ls", "-l", notebooks_folder])

    # Create the zip file from assignments folder and notebooks folder
    print(f"\n================= CREATING ZIP FILE =================\n")

    # Remove the root path for the zip command
    assignments_folder_relative = assignments_folder.replace(PROJECT_DIR + "/", "")
    notebooks_folder_relative = notebooks_folder.replace(PROJECT_DIR + "/", "")

    print(f"\tINFO: Assignments folder relative path: {assignments_folder_relative}")
    print(f"\tINFO: Notebooks folder relative path: {notebooks_folder_relative}")

    print(f"\tINFO: Creating {output_file}.")
    print(
        f"\tINFO: Adding {assignments_folder} and {notebooks_folder} to the zip file."
    )
    subprocess.run(["zip", "-r", output_file, assignments_folder_relative, notebooks_folder_relative],
                   cwd=PROJECT_DIR)

    # Remove the notebooks folder
    print(f"\n================= CLEANUP =================\n")
    if os.path.isdir(notebooks_folder):
        shutil.rmtree(notebooks_folder)


if __name__ == "__main__":
    submit()

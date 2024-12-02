import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import uuid
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

warnings.filterwarnings("ignore")
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def load_config(config_file_path):
    with open(config_file_path, "r") as file:
        config_file = yaml.safe_load(file)
    return config_file


def load_annotations_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df


def clone_pyBioCLIP_from_github(directory, repo_url):
    if not os.path.exists(directory) or not os.listdir(directory):
        print(f"Cloning repository {repo_url} into {directory}...")
        subprocess.run(["git", "clone", repo_url, directory], check=True)
        print("Repository cloned successfully...")
    else:
        print("Repository already cloned...")


# def install_pyBioCLIP_from_directory(directory):
#     try:
#         print(f"Installing package from {directory}...")
#         subprocess.run([sys.executable, '-m', 'pip', 'install', directory], check=True)
#         print("Package installed successfully...")
#     except subprocess.CalledProcessError as e:
#         print(f"Error installing package from {directory}: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")


def install_pyBioCLIP_from_directory(directory):
    try:
        print(f"Installing package from {directory}...")
        subprocess.run([sys.executable, "-m", "pip", "install", directory], check=True)
        print("Package installed successfully...")

        # Define the path to your custom predict.py file
        custom_predict_path = "predict.py"

        # Locate the installed package path
        # import bioclip  # Import to get the package's directory
        # package_path = os.path.dirname(bioclip.__file__)
        package_path = "bioclip"

        # Define the path to the target predict.py file in the installed package
        target_predict_path = os.path.join(package_path, "predict.py")

        # Copy the custom predict.py file to the installed package directory
        shutil.copy(custom_predict_path, target_predict_path)
        print(f"Replaced predict.py with custom file from {custom_predict_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error installing package from {directory}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_bioCLIP(url, target_dir):
    try:
        result = subprocess.run(
            ["pip", "install", f"git+{url}", "--target", target_dir],
            capture_output=True,
            check=True,
        )
        print(f"Successfully installed from {url} to {target_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing from {url} to {target_dir}:")
        print(e.stderr.decode("utf-8"))


def cache_BioCLIP(previous_file_path, updated_file_path):
    assert os.path.exists(previous_file_path) == True, f"ERROR!"
    assert os.path.exists(updated_file_path) == True, f"ERROR!"
    os.remove(previous_file_path)
    shutil.copy(updated_file_path, os.path.dirname(previous_file_path))


def run_pyBioclip(bioclip_classifier, image_dir, df):

    predicted_labels = []
    predicted_scores = []

    for index, row in df.iterrows():

        x0, y0, x1, y1 = ast.literal_eval(row["bbox pred"])
        image_filename = row["image uuid"] + ".jpg"

        image_filepath = os.path.join(image_dir, image_filename)
        original_image = Image.open(image_filepath)
        cropped_image = original_image.crop((x0, y0, x1, y1))

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.close()
        cropped_image.save(temp_file.name)

        predictions = bioclip_classifier.predict(temp_file.name)

        top_prediction = max(predictions, key=lambda x: x["score"])
        predicted_label = top_prediction["classification"]
        pred_conf_score = top_prediction["score"]

        predicted_labels.append(predicted_label)
        predicted_scores.append(pred_conf_score)
        os.remove(temp_file.name)

    df["species_prediction"] = predicted_labels
    df["species_pred_score"] = predicted_scores

    return df


def pyBioCLIP(labels, image_dir, df):

    classifier = CustomLabelsClassifier(labels)
    df = run_pyBioclip(classifier, image_dir, df)

    return df


def simplify_species(species_name, category_map):
    for key, value in category_map.items():
        if key in species_name:
            return value
    return None


def postprerocess_dataframe(df):

    category_map_true = {"zebra_grevys": 0, "zebra_plains": 1, "neither": 2}
    df["species_true_simple"] = df["annot species"].apply(
        lambda x: simplify_species(x, category_map_true)
    )

    category_map_pred = {"grevy's zebra": 0, "plains zebra": 1, "neither": 2}
    df["species_pred_simple"] = df["species_prediction"].apply(
        lambda x: simplify_species(x, category_map_pred)
    )

    return df


def plot_confusion_matrix(prediction_df, labels):

    true_labels = prediction_df["species_true_simple"]
    pred_labels = prediction_df["species_pred_simple"]

    conf_matrix = confusion_matrix(
        true_labels, pred_labels, labels=true_labels.unique()
    )
    conf_matrix_df = pd.DataFrame(
        conf_matrix, index=true_labels.unique(), columns=true_labels.unique()
    )

    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average="weighted")
    recall = recall_score(true_labels, pred_labels, average="weighted")
    f1 = f1_score(true_labels, pred_labels, average="weighted")

    class_accuracy = {}
    classes = true_labels.unique()
    for ix in classes:
        correct = conf_matrix_df.loc[ix, ix]
        total = np.sum(conf_matrix_df.loc[ix, :])
        class_accuracy[ix] = correct / total

    print("PERFORMANCE SUMMARY")
    print("=" * 40)
    print(f"Accuracy  : {accuracy:.2f}")
    print(f"Precision : {precision:.2f}")
    print(f"Recall    : {recall:.2f}")
    print(f"F-1 Score : {f1:.2f}")

    print("-" * 40)
    print("\nClass-wise Accuracy:")
    print("-" * 40)
    for ix in classes:
        print(f"Accuracy of {labels[ix]:<20}: {class_accuracy[ix]:.2f}")

    print("\n\nPlotting Confusion Matrix ...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":

    # Loading Configuration File ...
    config = load_config("species_identifier_drive.yaml")

    # Setting up Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(
        description="Detect bounding boxes for database of animal images"
    )
    parser.add_argument(
        "image_dir", type=str, help="The directory where localized images are found"
    )
    parser.add_argument(
        "in_csv_path",
        type=str,
        help="The full path to the viewpoint classifier output csv to use as input",
    )
    parser.add_argument(
        "si_dir", type=str, help="The directory to install bioCLIP within"
    )
    parser.add_argument(
        "out_csv_path", type=str, help="The full path to the output csv file"
    )
    args = parser.parse_args()

    images = Path(args.image_dir)

    if os.path.exists(args.si_dir):
        print("Removing Previous Instance of Experiment")
        shutil.rmtree(args.si_dir)

    print("Creating Experiment Directory ...")
    os.makedirs(args.si_dir, exist_ok=True)

    bioCLIP_dir = os.path.join(args.si_dir, config["bioclip_dirname"])
    pyBioCLIP_url = config["github_bioclip_url"]

    print("Cloning & Installing pyBioCLIP ...")
    clone_pyBioCLIP_from_github(bioCLIP_dir, pyBioCLIP_url)
    install_pyBioCLIP_from_directory(bioCLIP_dir)
    from bioclip import CustomLabelsClassifier

    print("pyBioCLIP Installation Completed ....")

    print("Caching pyBioCLIP ...")
    prev_predict_filepath = os.path.join(bioCLIP_dir, config["org_predict_file"])
    new_predict_filepath = Path(config["new_predict_file"])
    cache_BioCLIP(prev_predict_filepath, new_predict_filepath)
    print("Caching Completed ...")

    print("Running pyBioCLIP ...")
    labels = config["custom_labels"]
    df = load_annotations_from_csv(args.in_csv_path)
    df = pyBioCLIP(labels, images, df)
    print("pyBioCLIP Completed ...")

    print("Post-Processing ...")
    df = postprerocess_dataframe(df)
    print("Post-Processing Completed ...")

    print("Showing Results and Plotting Confusion Matrix ...")
    plot_confusion_matrix(df, labels)
    print("Confusion Matrix Plotted ...")

    prediction_dir = os.path.dirname(args.out_csv_path)

    print("Saving ALL Predictions as CSV ...")

    df.to_csv(args.out_csv_path, index=False)

    print("Completed Successfully!")

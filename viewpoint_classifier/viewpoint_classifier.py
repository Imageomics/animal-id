import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset
import timm
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import os
import shutil
import ast, warnings
import argparse
import seaborn as sns
from PIL import Image, ImageDraw


# Load configuration
with open('./viewpoint_classifier.yaml', 'r') as file:
    config = yaml.safe_load(file)


class ClassifierDataset(Dataset):
    def __init__(self, df, transforms=None, output_label=False):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms

        self.output_label = output_label
        #self.label_cols = label_cols

        if self.output_label:
            # Aggregate the label columns into a single multi-hot encoded vector
            self.labels = self.df[self.label_cols].values  # This creates a NumPy array of shape [num_samples, num_labels]
            self.labels = torch.tensor(self.labels, dtype=torch.float32)  # Convert to a tensor for PyTorch compatibility

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img = get_chip(self.df.loc[index])
        #print(f'Shape of the input image: {img.shape}')    # Print the shape of the image
        if self.transforms:
            img = self.transforms(image=img)['image']  # Apply transformations
            #print(f'Shape of the transformed image: {img.shape}')
        if self.output_label:
            # Load label data
            target = self.labels[index]
            return img, target
        else:
            return img

    def load_image(self, img_path):
        # Load image from the file system; placeholder function
        # You should replace this with actual image loading logic
        img = np.random.rand(224, 224, 3)  # Placeholder: Replace with actual image loading
        return img

class ImgClassifier(torch.nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x

def get_valid_transforms():
    return Compose([
            Resize(config['img_size'], config['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


def predict_labels_new(test_loader, model, device):
    model.eval()

    # Store predictions and discrete labels for all samples
    all_preds = []
    all_discrete_labels = []

    with torch.no_grad():
        for imgs in test_loader:
            imgs = imgs.to(device).float()

            # Make the prediction
            image_preds = model(imgs)
            preds_sigmoid = torch.sigmoid(image_preds)  # Apply sigmoid to get probabilities
            all_preds.append(preds_sigmoid.detach().cpu())

            # Convert probabilities to labels based on a threshold
            threshold = 0.5
            discrete_labels = (preds_sigmoid > threshold).int()
            all_discrete_labels.append(discrete_labels.detach().cpu())

    # Concatenate all batch results
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_discrete_labels = torch.cat(all_discrete_labels, dim=0).numpy()

    return all_preds, all_discrete_labels


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb

def rotate_box(x1,y1,x2,y2,theta):
    xm = (x1 + x2) // 2
    ym = (y1 + y2) // 2
    h = int(y2 - y1)
    w = int(x2 - x1)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    A = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]])
    C = np.array([[xm, ym]])
    RA = (A - C) @ R.T + C
    RA = RA.astype(int)
    return RA

def crop_rect(img, rect):
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    height, width = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D(center, np.rad2deg(angle), 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop, img_rot

def get_chip(row):
    #box = ast.literal_eval(row['bbox'])
    box = row['bbox']
    theta = 0.0
    img = get_img(row['path']).copy()
    x1,y1,w,h = box
    x2 = x1 + w
    y2 = y1 + h
    xm = (x1 + x2) // 2
    ym = (y1 + y2) // 2
    return crop_rect(img, ((xm, ym), (x2-x1, y2-y1), theta))[0]


def bbox_match(gt_bbox, pred_bbox, tolerance=10):
    """
    Check if bounding boxes match within a tolerance
    Handles different bbox formats (x1,y1,x2,y2 vs x,y,w,h)
    """
    try:
        # Convert to list if string
        if isinstance(gt_bbox, str):
            gt_bbox = ast.literal_eval(gt_bbox)

        # If pred_bbox is a tuple from zip, convert to list
        pred_bbox = list(pred_bbox)

        # Convert pred_bbox from [x,y,w,h] to [x1,y1,x2,y2]
        pred_bbox_converted = [
            pred_bbox[0],  # x1
            pred_bbox[1],  # y1
            pred_bbox[0] + pred_bbox[2],  # x2 = x + width
            pred_bbox[1] + pred_bbox[3]   # y2 = y + height
        ]

        return all(abs(gt - pred) <= tolerance for gt, pred in zip(gt_bbox, pred_bbox_converted))
    except Exception as e:
        print(f"Error in bbox_match: {e}")
        return False





def save_misclassified_images(predictions_csv, ground_truth_csv, output_dir):
    """
    Save misclassified images for each label as one large image with 3 images per row,
    cropping each image based on its bounding box.
    """
    # Read prediction and ground truth CSVs
    pred_df = pd.read_csv(predictions_csv)
    gt_df = pd.read_csv(ground_truth_csv)

    # Clean viewpoints for uniformity
    def clean_viewpoint(viewpoint):
        if pd.isna(viewpoint):
            return ['unknown']
        viewpoints = [v.strip() for v in str(viewpoint).lower().split(',')]
        return viewpoints

    # Match predictions to ground truth
    matched_rows = []
    for idx, gt_row in gt_df.iterrows():
        try:
            matching_preds = pred_df[(pred_df['image uuid'] == gt_row['filename']) & (pred_df['annot species'] == config['species'])]
            for _, pred_row in matching_preds.iterrows():
                if bbox_match(gt_row['bbox_x'],
                              (pred_row['bbox x'], pred_row['bbox y'], pred_row['bbox w'], pred_row['bbox h'])):
                    matched_rows.append({
                        'image_uuid': pred_row['image uuid'],
                        'predicted_viewpoint': pred_row['predicted_viewpoint'],
                        'ground_truth_viewpoint': gt_row['viewpoint'],
                        'bbox_x': pred_row['bbox x'],
                        'bbox_y': pred_row['bbox y'],
                        'bbox_w': pred_row['bbox w'],
                        'bbox_h': pred_row['bbox h'],
                        'path': pred_row['path']  # Include image path
                    })
        except Exception:
            continue

    # Convert matches to DataFrame
    if not matched_rows:
        print("No matches found!")
        return
    result_df = pd.DataFrame(matched_rows)

    # Clean viewpoints
    result_df['predicted_viewpoint_clean'] = result_df['predicted_viewpoint'].apply(clean_viewpoint)
    result_df['ground_truth_viewpoint_clean'] = result_df['ground_truth_viewpoint'].apply(clean_viewpoint)

    # Define labels
    labels = sorted(['right', 'left', 'up', 'down', 'front', 'back'])

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process misclassifications
    for label in labels:
        misclassifications = result_df[
            result_df['ground_truth_viewpoint_clean'].apply(lambda x: label in x) !=
            result_df['predicted_viewpoint_clean'].apply(lambda x: label in x)
        ]

        if misclassifications.empty:
            print(f"No misclassifications for label '{label}'.\n")
            continue

        # Display up to 10 misclassified images
        count = min(10, len(misclassifications))
        print(f"Saving misclassified images for label '{label}'...")

        # Create a blank canvas
        image_width, image_height = 200, 200  # Size of each cropped image
        images_per_row = 3
        rows = (count + images_per_row - 1) // images_per_row
        canvas_width = images_per_row * image_width
        canvas_height = rows * image_height

        canvas = Image.new('RGB', (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(canvas)

        for i in range(count):
            img_path = misclassifications.iloc[i]['path']
            bbox_x = misclassifications.iloc[i]['bbox_x']
            bbox_y = misclassifications.iloc[i]['bbox_y']
            bbox_w = misclassifications.iloc[i]['bbox_w']
            bbox_h = misclassifications.iloc[i]['bbox_h']
            true_labels = ', '.join(misclassifications.iloc[i]['ground_truth_viewpoint_clean'])
            predicted_labels = ', '.join(misclassifications.iloc[i]['predicted_viewpoint_clean'])

            try:
                img = Image.open(img_path)

                # Crop the image using the bounding box
                cropped_img = img.crop((
                    int(bbox_x),
                    int(bbox_y),
                    int(bbox_x + bbox_w),
                    int(bbox_y + bbox_h)
                )).resize((image_width, image_height))

                x_offset = (i % images_per_row) * image_width
                y_offset = (i // images_per_row) * image_height

                canvas.paste(cropped_img, (x_offset, y_offset))
                draw.text(
                    (x_offset + 5, y_offset + 5),
                    f"T: {true_labels}\nP: {predicted_labels}",
                    fill="black"
                )
            except Exception as e:
                print(f"Could not process image at {img_path}: {e}")

        # Save the canvas
        output_path = os.path.join(output_dir, f"misclassifications_{label}.png")
        canvas.save(output_path)
        print(f"Saved misclassified images for label '{label}' to {output_path}")





def evaluate_viewpoint_classification_with_visuals(predictions_csv, ground_truth_csv):
    """
    Evaluate viewpoint classification, calculate label-wise accuracies, confusion matrices, and display misclassifications.
    """
    pred_df = pd.read_csv(predictions_csv)
    gt_df = pd.read_csv(ground_truth_csv)

    def clean_viewpoint(viewpoint):
        if pd.isna(viewpoint):
            return 'unknown'
        viewpoints = [v.strip() for v in str(viewpoint).lower().split(',')]
        return ', '.join(viewpoints)

    # Match predictions to ground truth
    matched_rows = []
    for idx, gt_row in gt_df.iterrows():
        try:
            matching_preds = pred_df[(pred_df['image uuid'] == gt_row['filename']) & (pred_df['annot species'] == config['species'])]
            for _, pred_row in matching_preds.iterrows():
                if bbox_match(gt_row['bbox_x'],
                              (pred_row['bbox x'], pred_row['bbox y'], pred_row['bbox w'], pred_row['bbox h'])):
                    matched_rows.append({
                        'image_uuid': pred_row['image uuid'],
                        'predicted_viewpoint': pred_row['predicted_viewpoint'],
                        'ground_truth_viewpoint': gt_row['viewpoint'],
                        'bbox_x': pred_row['bbox x'],
                        'bbox_y': pred_row['bbox y'],
                        'bbox_w': pred_row['bbox w'],
                        'bbox_h': pred_row['bbox h'],
                        'path': pred_row['path']
                    })
        except Exception:
            continue

    # Convert matches to DataFrame
    if not matched_rows:
        print("No matches found!")
        return None, None
    result_df = pd.DataFrame(matched_rows)

    # Clean viewpoints
    result_df['predicted_viewpoint_clean'] = result_df['predicted_viewpoint'].apply(clean_viewpoint)
    result_df['ground_truth_viewpoint_clean'] = result_df['ground_truth_viewpoint'].apply(clean_viewpoint)

    labels = sorted(['right', 'left', 'up', 'down', 'front', 'back'])
    result_df['labels_bin'] = result_df['ground_truth_viewpoint_clean'].apply(
        lambda x: [int(label in x.split(', ')) for label in labels])
    result_df['predicted_viewpoint_bin'] = result_df['predicted_viewpoint_clean'].apply(
        lambda x: [int(label in x.split(', ')) for label in labels] if pd.notna(x) else [0] * len(labels))


    # Calculate accuracy for each label
    accuracies = {}
    for i, label in enumerate(labels):
        accuracies[label] = accuracy_score(
            result_df['labels_bin'].apply(lambda x: x[i]),
            result_df['predicted_viewpoint_bin'].apply(lambda x: x[i])
        )
    print("Label-wise Accuracies:", accuracies)

    # Calculate confusion matrix for each label
    for i, label in enumerate(labels):
        true_labels = result_df['labels_bin'].apply(lambda x: x[i])
        predicted_labels = result_df['predicted_viewpoint_bin'].apply(lambda x: x[i])
        cm = confusion_matrix(true_labels, predicted_labels)

        print(f"\nFor viewpoint '{label}':")
        print("Confusion Matrix:")
        print(cm)

        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            print(f"True label 1 predicted as 1 (True Positives): {tp}")
            print(f"True label 1 predicted as 0 (False Negatives): {fn}")
            print(f"True label 0 predicted as 1 (False Positives): {fp}")
            print(f"True label 0 predicted as 0 (True Negatives): {tn}\n")
        else:
            print("Not enough data to calculate performance metrics for this viewpoint.\n")

    # Overall confusion matrix
    unique_labels = sorted(set(result_df['ground_truth_viewpoint_clean'].unique()) |
                           set(result_df['predicted_viewpoint_clean'].unique()))
    cm = confusion_matrix(result_df['ground_truth_viewpoint_clean'],
                          result_df['predicted_viewpoint_clean'],
                          labels=unique_labels)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Viewpoint Classification Confusion Matrix')
    plt.xlabel('Predicted Viewpoint')
    plt.ylabel('Ground Truth Viewpoint')
    plt.tight_layout()
    plt.savefig('viewpoint_confusion_matrix.png')
    plt.close()

    # Classification report
    report = classification_report(result_df['ground_truth_viewpoint_clean'],
                                   result_df['predicted_viewpoint_clean'],
                                   labels=unique_labels)
    with open('classification_report.txt', 'w') as f:
        f.write(report)

    # Save the results
    result_df.to_csv('merged_viewpoint_data.csv', index=False)
    print("\nMerged DataFrame and additional metrics saved successfully.")

    return cm, report, accuracies



import argparse

if __name__ == "__main__":
    print("Loading data...")
    parser = argparse.ArgumentParser(description='Run viewpoint classifier for database of animal images')
    parser.add_argument('image_dir', type=str, help='The directory where localized images are found')
    parser.add_argument('in_csv_path', type=str, help='The full path to the viewpoint classifier output CSV to use as input')
    parser.add_argument('out_csv_path', type=str, help='The full path to the output CSV file')
    parser.add_argument('--model_checkpoint_path', type=str, default="viewpoint_trained_model.pth",
                        help='The full path to the model checkpoint (default: viewpoint_trained_model.pth)')
    parser.add_argument('--gt_csv_path', type=str, default=None,
                        help='The full path to the ground truth CSV (optional)')

    args = parser.parse_args()

    if args.gt_csv_path:
        print(f"Ground truth CSV path provided: {args.gt_csv_path}")
    else:
        print("No ground truth CSV path provided. Proceeding without it.")

    print(f"Using model checkpoint: {args.model_checkpoint_path}")


    original_csv = pd.read_csv(args.in_csv_path)

    # Append image_dir to the 'image fname' column
    original_csv['path'] = original_csv['image fname'].apply(lambda x: os.path.join(args.image_dir, x))

    # Create a single 'bbox' column from the four bbox columns
    original_csv['bbox'] = list(
        zip(original_csv['bbox x'], original_csv['bbox y'], original_csv['bbox w'], original_csv['bbox h']))

    # Split the original dataframe
    filtered_test = original_csv[
        (original_csv[['bbox x', 'bbox y', 'bbox w', 'bbox h']].notna().all(axis=1)) &
        (original_csv['annot species'] == config['species'])
        ].reset_index(drop=True)
    other_test = original_csv[
        ~(original_csv[['bbox x', 'bbox y', 'bbox w', 'bbox h']].notna().all(axis=1)) |
        (original_csv['annot species'] != config['species'])
        ].reset_index(drop=True)
    other_test['predicted_viewpoint'] = np.nan

    print("Preparing data for the model...")
    test_ds = ClassifierDataset(filtered_test, transforms=get_valid_transforms())
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config['valid_bs'],
        num_workers=config['num_workers'],
        shuffle=False,
        pin_memory=False
    )

    print("Setting up the model...")
    device = torch.device(config['device'])
    with warnings.catch_warnings():  # Add this line
        warnings.filterwarnings("ignore", category=UserWarning)
        model = ImgClassifier(config['model_arch'], len(config['label_cols']), pretrained=True).to(device)
        model.load_state_dict(torch.load(args.model_checkpoint_path, map_location=config['device']))
        scaler = GradScaler()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config['T_0'], T_mult=1, eta_min=float(config['min_lr']), last_epoch=-1)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)

    print("Running the model...")
    _, all_discrete_labels = predict_labels_new(test_loader, model, device)

    print("Processing the model predictions...")
    preds_bin = pd.DataFrame(all_discrete_labels, columns=config['label_cols'])
    filtered_test['predicted_viewpoint'] = preds_bin.apply(lambda row: ', '.join(row.index[row == 1]), axis=1)

    # Concatenate results
    final_output = pd.concat([filtered_test, other_test])

    # Save results
    output_file = args.out_csv_path  # Full path to the output file
    if os.path.exists(output_file):
        print(f"Removing Previous Instance of File: {output_file}")
        os.remove(output_file)

    print("Saving the results...")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Save the DataFrame to the specified file
    final_output.to_csv(output_file, index=False)

    if args.gt_csv_path:
        evaluate_viewpoint_classification_with_visuals(args.out_csv_path, args.gt_csv_path)
        save_misclassified_images(args.out_csv_path, args.gt_csv_path, "misclassified_images")

    print("Done!")
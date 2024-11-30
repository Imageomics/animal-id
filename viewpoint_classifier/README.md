## Viewpoint Classifier Script

### Overview
This script processes animal image data and predicts their viewpoints using a pre-trained model.

### Usage
Run the script from the command line with the following arguments:
```bash
python viewpoint_classifier.py <image_dir> <in_csv_path> <out_csv_path> [model_checkpoint_path] [gt_csv_path]
```

### Arguments
1. **`image_dir`**: Directory containing localized images.
2. **`in_csv_path`**: Path to the input CSV file for the classifier.
3. **`out_csv_path`**: Path to save the output CSV file.
4. **`model_checkpoint_path`** *(optional)*: Path to the model checkpoint file. (use default if not provided)
5. **`gt_csv_path`** *(optional)*: Path to the ground truth CSV file for evaluation.

### Example
```bash
python viewpoint_classifier.py ../input_img_example ./input_example.csv ./output_example.csv
```

### Output
1. Processed CSV file at the specified output path.
2. If `gt_csv_path` is provided:
   - Visual evaluation metrics.
   - Misclassified images saved in the `misclassified_images` directory.

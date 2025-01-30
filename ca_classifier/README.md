## Census Annotation Classifier Script

### Overview
This script gives a binary classification for census annotation on grevy's zebra detections. 

### Usage
Run the script from the command line with the following arguments:
```bash
python CA_classifier.py <image_dir> <in_csv_path> <cac_model_path> <out_csv_path>
```
### Arguments
1. **`image_dir`**: Directory containing localized images.
2. **`in_csv_path`**: The full path to the ca classifier input csv.
3. **`cac_model_path`**: The full path to the ca classifier model.
4. **`out_csv_path`**: The full path to the output csv file.
### Example
```bash
python CA_classifier.py ../input_img_example ../viewpoint_classifier/output_example.csv ./CA_trained_model.pth ./annots/output.csv
```
### Output
Processed CSV and JSON file at `annots/output.csv`.
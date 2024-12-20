## Detector Script

### Overview
This script processes animal image data and detects animal bounding boxes using a pre-trained model. Human detections are ignored.

### Usage
Run the script from the command line with the following arguments:
```bash
python detector.py <image_dir> <in_csv_path> <si_dir> <out_csv_path>
```

### Arguments
1. **`image_dir`**: Directory containing localized images.
2. **`in_csv_path`**: The full path to the viewpoint classifier output csv.
3. **`si_dir`**: The directory to install bioCLIP within.
4. **`out_csv_path`**: The full path to the output csv file.
### Example
```bash
python species_identifier.py ../input_img_example ../detector/annots/annots.csv ./bioclip ./output 
```
### Output
Processed CSV and JSON file at `annot_dir/annots_filename.*`.

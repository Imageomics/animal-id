## Detector Script

### Overview
This script processes animal image data and detects animal bounding boxes using a pre-trained model. Human detections are ignored.

### Usage
Run the script from the command line with the following arguments:
```bash
python detector.py <image_dir> <annot_dir> <exp_dir> <model_version> <annots_filename>
```

### Arguments
1. **`image_dir`**: Directory containing localized images.
2. **`annot_dir`**: Directory containing output annotations.
3. **`exp_dir`**: Directory to export models and predictions to.
4. **`model version`**: The Yolo model version to use.
5. **`annots_filename`**: Name of annotations file(s).
### Example
```bash
python detector.py ../input_img_example ./annots ./detector yolov10l annots
```
### Output
Processed CSV and JSON file at `annot_dir/annots_filename.*`.

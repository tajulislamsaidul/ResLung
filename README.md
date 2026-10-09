# LungCANet

**LungCANet 4.1.0** is a Python desktop application for research and educational analysis of lung CT images using AI models.

> **Disclaimer:** This project is for research and educational use only. It is not a medical diagnosis tool and must not replace a qualified healthcare professional.

## Features

- Classifies images as **Benign**, **Malignant**, or **Normal**
- Preprocesses images before analysis
- Displays prediction scores and confidence levels
- Generates Grad-CAM visualizations when supported by the model
- Compares models and supports ensemble predictions
- Supports batch image analysis
- Saves analysis history in SQLite
- Exports results to PDF, JSON, CSV, and Excel (some exports require optional packages)
- Provides light and dark themes

## Technologies

- Python
- Tkinter
- TensorFlow / Keras
- OpenCV
- NumPy
- Pillow
- SQLite
- ReportLab (PDF reports)
- openpyxl (Excel exports)

## Installation

1. Install Python and make sure Tkinter is available.
2. Install the required packages:

   ```bash
   pip install tensorflow opencv-python numpy Pillow reportlab openpyxl
   ```

3. Create a `models` folder beside the Python file.
4. Place your trained `.keras` or `.h5` model files inside `models`.

**Important:** The project code does not include trained model files. The model must return three outputs in this order: `Benign`, `Malignant`, `Normal`. Images are resized to `224 × 224`.

## Run the application

Save the application code as `lungcanet.py`, then run:

```bash
python lungcanet.py
```

## How to use

1. Select a model.
2. Click **Open CT Image** and choose an image.
3. Click **Analyze CT Image**.
4. Review the predicted class, confidence, class scores, and visualizations.
5. Use model comparison, ensemble prediction, batch analysis, or export options if needed.

Supported image formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`. DICOM files are not supported by the provided code.

## Expected output

The app displays one of three predicted classes, a confidence score, scores for all three classes, and a Grad-CAM overlay when available. It can also save reports and analysis history.

## Output folders

The application creates an `outputs` folder containing analysis history, reports, batch results, and a SQLite database.

## Future improvements

- Add tests and documented dependency versions.
- Validate model input settings and class order.
- Add DICOM support if needed.
- Add tools to evaluate model performance on labeled test data.
- Improve privacy and error-handling features.


**Research and educational use only.**

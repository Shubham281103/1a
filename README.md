# PDF Outline Extractor

## 1. Project Overview

This project provides a complete, dockerized pipeline to automatically extract a structured outline from a collection of PDF documents. It processes each PDF, identifies the title and headings (H1, H2, H3), and generates a structured JSON file for each document, detailing its hierarchical structure.

The system is designed to be robust, handling both text-based and image-based (scanned) PDFs.

## 2. How It Works

The solution is built on a two-stage process: intelligent PDF parsing followed by machine learning classification.

### Stage 1: Hybrid PDF Parsing

The core of the project is a sophisticated parsing engine within the `predict.py` script that intelligently handles different types of PDFs.

* **Hybrid Approach:** For each page in a PDF, the script first attempts to extract text directly. If it finds very little selectable text (a sign of a scanned document), it automatically switches to an OCR (Optical Character Recognition) fallback.
* **OCR Pre-processing:** For scanned pages, images are first cleaned using OpenCV to improve OCR accuracy. This involves:
    * Converting the image to grayscale.
    * Applying an adaptive threshold to create a clean, high-contrast black and white image.
* **Text Merging:** After initial extraction, a post-processing step merges text fragments that appear on the same horizontal line but are separated by large spaces (e.g., "2.2" and "Career Paths for Testers"). This ensures that headings are captured as single, coherent lines.

### Stage 2: Machine Learning Classification

Once the text is cleanly extracted, a trained Random Forest model classifies each line.

* **Model:** A `RandomForestClassifier` from the `scikit-learn` library.
* **Feature Engineering:** The model doesn't just use basic text properties. It relies on a set of engineered features to make intelligent decisions:
    * **Base Features:** `font_size`, `is_bold`, and positional coordinates (`x0`, `y0`, `x1`, `y1`).
    * **Engineered Features:**
        * `relative_font_size`: How large a font is compared to the most common font on its page. This is a powerful indicator for headers.
        * `line_width`: The horizontal width of the text block.
        * `normalized_y_pos`: The vertical position on the page (0.0 = top, 1.0 = bottom), useful for identifying titles and footers.
* **Data Balancing:** The model was trained on a balanced dataset created using **random under-sampling**. This technique prevents the model from being biased towards the most common class (body text) and significantly improves its ability to detect the rarer heading classes.

## 3. Directory Structure

To run the project, you must organize your files in the following structure:

```
/your_project_folder/
|
|-- pdfs/
|   |-- document1.pdf
|   |-- document2.pdf
|   |-- ... (place all PDFs to be analyzed here)
|
|-- model/
|   |-- document_classifier.joblib  (the trained model)
|   |-- label_encoder.joblib      (the label encoder)
|
|-- output/
|   |-- (this folder will be created automatically)
|
|-- predict.py
|-- requirements.txt
|-- Dockerfile
```

## 4. Prerequisites

* **Docker Desktop:** The entire application is containerized, so you only need Docker installed and running on your machine.

## 5. Steps of Execution

Follow these steps from your terminal inside `your_project_folder`.

### Step 1: Build the Docker Image

This command packages your script and all its dependencies into a self-contained image named `final-predictor`. You only need to run this once, or whenever you change the code.

```bash
docker build -t final-predictor .
```

### Step 2: Run the Prediction Container

This command runs the prediction script on all the PDFs in your `pdfs` folder. It uses a temporary container to process the files.

```bash
docker run --name predictor-instance -v "./pdfs:/app/pdfs" -v "./model:/app/model" final-predictor
```

### Step 3: Copy the Results Manually

Due to a common file-syncing issue with Docker on Windows, the output folder may not appear automatically. Use this command to manually copy the generated `output` folder from the container to your local directory.

```bash
docker cp predictor-instance:/app/output .
```

### Step 4: Clean Up the Container

After copying your results, remove the temporary container to keep your system clean.

```bash
docker rm predictor-instance
```

## 6. Output Format

After running the process, the `output` folder will contain one JSON file for each PDF you processed. The format of each JSON file is as follows:

```json
{
    "title": "The Main Title of the Document",
    "outline": [
        {
            "level": "H1",
            "text": "This is a top-level heading",
            "page": 0
        },
        {
            "level": "H2",
            "text": "This is a sub-heading",
            "page": 1
        },
        {
            "level": "H3",
            "text": "This is a sub-sub-heading",
            "page": 1
        }
    ]
}
```

**Note:** Page numbers are **0-indexed** as requested.

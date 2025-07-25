import pandas as pd
import joblib
import os
import glob
import re
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import numpy as np
import cv2 # OpenCV library
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ==============================================================================
#  STEP 1: REUSE THE EXACT SAME PARSING AND FEATURE ENGINEERING LOGIC
# ==============================================================================

def process_page_with_ocr(page: fitz.Page) -> list:
    """Fallback to OCR with image pre-processing for image-based pages."""
    features = []
    try:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_cv = np.array(img)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        custom_config = r'--oem 3 --psm 3'
        ocr_data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT, config=custom_config)
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            if text and int(ocr_data['conf'][i]) > 50:
                features.append({
                    "text": text, "x0": ocr_data['left'][i], "y0": ocr_data['top'][i],
                    "x1": ocr_data['left'][i] + ocr_data['width'][i], "y1": ocr_data['top'][i] + ocr_data['height'][i],
                    "font_size": 12.0, "font_name": "OCR", "is_bold": False,
                })
    except Exception as e:
        print(f"      - OCR processing failed with error: {e}")
    return features

def extract_features_from_pdf(pdf_path: str) -> list:
    """Extracts text using native text extraction with an OCR fallback."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"    - Could not open '{os.path.basename(pdf_path)}'. Error: {e}")
        return []

    all_features = []
    for page_num, page in enumerate(doc):
        page_features = []
        if len(page.get_text().strip()) < 100: 
            print(f"      - Page {page_num+1} seems image-based, falling back to OCR...")
            page_features = process_page_with_ocr(page)
        else:
            page_dict = page.get_text("dict", sort=True)
            for block in page_dict.get("blocks", []):
                if "lines" in block:
                    for line in block.get("lines", []):
                        if line.get("spans"):
                            line_text = " ".join([s["text"].strip() for s in line["spans"]])
                            if line_text:
                                first_span = line["spans"][0]
                                page_features.append({
                                    "text": line_text, "font_size": round(first_span["size"], 2),
                                    "font_name": first_span["font"], "is_bold": "bold" in first_span["font"].lower(),
                                    "x0": round(line["bbox"][0], 2), "y0": round(line["bbox"][1], 2),
                                    "x1": round(line["bbox"][2], 2), "y1": round(line["bbox"][3], 2),
                                })
        
        for feature in page_features:
            feature["filename"] = os.path.basename(pdf_path)
            feature["page_num"] = page_num + 1
        all_features.extend(page_features)
    return all_features

def merge_horizontally(df: pd.DataFrame, y_tolerance: int = 3) -> pd.DataFrame:
    """Merges text fragments that are on the same horizontal line."""
    if df.empty: return df
    merged_rows = []
    df_sorted = df.sort_values(by=['filename', 'page_num', 'y0', 'x0']).reset_index(drop=True)
    df_sorted['line_group'] = (df_sorted['y0'] // y_tolerance).astype(int)
    grouped = df_sorted.groupby(['filename', 'page_num', 'line_group'])
    for _, group in grouped:
        if len(group) == 1:
            merged_rows.append(group.iloc[0].to_dict())
            continue
        merged_row = group.iloc[0].to_dict()
        for i in range(1, len(group)):
            next_row = group.iloc[i]
            merged_row['text'] += ' ' + next_row['text']
            merged_row['x1'] = max(merged_row['x1'], next_row['x1'])
        merged_rows.append(merged_row)
    if not merged_rows: return pd.DataFrame()
    return pd.DataFrame(merged_rows).drop(columns=['line_group'])

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates the same features used during training."""
    print("    - Engineering features for prediction...")
    page_stats = df.groupby(['filename', 'page_num']).agg(
        page_height=('y1', 'max'),
        common_font_size=('font_size', lambda x: x.mode().iloc[0] if not x.mode().empty else 12.0)
    ).reset_index()
    df = pd.merge(df, page_stats, on=['filename', 'page_num'], how='left')
    df['relative_font_size'] = df['font_size'] / df['common_font_size'].replace(0, 1)
    df['line_width'] = df['x1'] - df['x0']
    df['normalized_y_pos'] = df['y0'] / df['page_height'].replace(0, 1)
    return df

# ==============================================================================
#  STEP 2: PREDICTION AND JSON FORMATTING
# ==============================================================================

def predict_and_generate_outline(
    pdf_path: str, 
    model: RandomForestClassifier, 
    encoder: LabelEncoder
) -> dict:
    """
    Processes a single PDF, predicts labels, and formats the output as a JSON structure.
    """
    features_data = extract_features_from_pdf(pdf_path)
    if not features_data:
        return {"title": "Could not extract text.", "outline": []}
    
    df = pd.DataFrame(features_data)
    df_merged = merge_horizontally(df)
    df_engineered = engineer_features(df_merged)

    features_for_prediction = [
        'font_size', 'is_bold', 'x0', 'y0', 'x1', 'y1',
        'relative_font_size', 'line_width', 'normalized_y_pos'
    ]
    for col in features_for_prediction:
        if col not in df_engineered:
            df_engineered[col] = 0
            
    X_new = df_engineered[features_for_prediction].copy()
    X_new['is_bold'] = X_new['is_bold'].astype(int)

    predictions_encoded = model.predict(X_new)
    predictions = encoder.inverse_transform(predictions_encoded)
    df_engineered['predicted_label'] = predictions

    output_json = {"title": "", "outline": []}
    
    title_df = df_engineered[df_engineered['predicted_label'] == 'title']
    if not title_df.empty:
        output_json["title"] = title_df.iloc[0]['text']
    
    headings_df = df_engineered[df_engineered['predicted_label'].isin(['H1', 'H2', 'H3'])]
    
    for _, row in headings_df.iterrows():
        output_json["outline"].append({
            "level": row['predicted_label'],
            "text": row['text'],
            # --- MODIFIED LINE: Changed to 0-based indexing ---
            "page": int(row['page_num']) - 1 
        })
        
    return output_json

def main():
    """
    Main function to run the batch prediction process.
    """
    pdf_folder = "pdfs"
    model_folder = "model"
    output_folder = "output"
    
    try:
        model = joblib.load(os.path.join(model_folder, "document_classifier.joblib"))
        encoder = joblib.load(os.path.join(model_folder, "label_encoder.joblib"))
        print("✅ Model and encoder loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Model files not found in '{model_folder}' directory. Please ensure they are present.")
        return
        
    os.makedirs(output_folder, exist_ok=True)
    
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"❌ No PDF files found in the '{pdf_folder}' directory.")
        return

    print(f"Found {len(pdf_files)} PDFs to process...")
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\nProcessing '{filename}'...")
        
        outline_data = predict_and_generate_outline(pdf_path, model, encoder)
        
        output_filename = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(output_folder, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(outline_data, f, indent=4, ensure_ascii=False)
            
        print(f"  -> ✅ Outline saved to '{output_path}'")
        os.chmod(output_path, 0o777)

if __name__ == "__main__":
    main()

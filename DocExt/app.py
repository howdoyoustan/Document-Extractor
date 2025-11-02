import streamlit as st
import sqlite3
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import io
import os
import easyocr
from ultralytics import YOLO
import numpy as np

# --- Configuration ---
MODEL_PATH = "best_model.pt"  # Path to your trained 'best.pt' file
DB_NAME = "customers.db"
CLASS_NAMES = ['Address', 'Class', 'DOB', 'Exp date', 'Face', 'First name', 'Issue date', 'Last name', 'License number', 'Sex']

# --- Database Functions ---

def init_db():
    """Initializes the SQLite database and creates the customers table."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Create table - dynamically creating columns based on class names (excluding 'Face')
    columns = [f'"{name.replace(" ", "_").lower()}" TEXT' for name in CLASS_NAMES if name != 'Face']
    create_table_query = f"CREATE TABLE IF NOT EXISTS customers (id INTEGER PRIMARY KEY, {', '.join(columns)})"
    c.execute(create_table_query)
    conn.commit()
    conn.close()

def add_customer(data):
    """Adds a new customer to the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Prepare data for insertion
    keys = []
    values = []
    for name, value in data.items():
        if name != 'Face':
            keys.append(f'"{name.replace(" ", "_").lower()}"')
            values.append(value)
            
    placeholders = ', '.join(['?'] * len(values))
    columns = ', '.join(keys)
    
    try:
        query = f"INSERT INTO customers ({columns}) VALUES ({placeholders})"
        c.execute(query, tuple(values))
        conn.commit()
        st.success("Customer successfully saved to the database!")
    except sqlite3.Error as e:
        st.error(f"An error occurred while saving to the database: {e}")
    finally:
        conn.close()

# --- Model & Processing Functions ---

@st.cache_resource
def load_models():
    """Loads the YOLOv8 and EasyOCR models into memory."""
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        st.error(f"YOLO model file not found at {MODEL_PATH}")
        st.stop()
        
    model = YOLO(MODEL_PATH)
    # Initialize EasyOCR reader
    # 'en' is for English. Add other languages if needed, e.g., ['en', 'es']
    reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if you have a compatible GPU
    return model, reader

def pdf_to_image(pdf_bytes):
    """Converts the first page of a PDF to a PIL Image."""
    try:
        # Open the PDF from bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc.load_page(0)  # Load the first page
        
        # Render page to a pixmap (image)
        # We can increase DPI for better quality
        pix = page.get_pixmap(dpi=300)
        
        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception as e:
        st.error(f"Error converting PDF: {e}")
        return None

def process_image(image, yolo_model, ocr_reader):
    """
    Runs YOLO detection and OCR on the image.
    Returns a dictionary of extracted data and the face image.
    """
    # --- 1. YOLOv8 Detection ---
    results = yolo_model.predict(image, conf=0.4) # Use a confidence threshold
    
    extracted_data = {}
    face_image = None
    
    # Convert PIL Image to NumPy array for cropping (OpenCV format)
    image_np = np.array(image)
    
    # Get the bounding boxes, class IDs, and class names
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    
    if len(boxes) == 0:
        st.warning("YOLO model did not detect any fields. Try a clearer image or adjust model confidence.")
        return {}, None

    # --- 2. Crop, OCR, and Process Results ---
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cls_name = CLASS_NAMES[class_ids[i]]
        
        # Crop the detected region
        cropped_image = image_np[y1:y2, x1:x2]
        
        if cls_name == 'Face':
            # --- 3a. Handle Face ---
            # Convert back to PIL Image for display
            face_image = Image.fromarray(cropped_image)
            extracted_data[cls_name] = face_image
            
        else:
            # --- 3b. Handle Text (OCR) ---
            # Use EasyOCR to read text from the cropped image
            # 'detail=0' returns a list of strings
            # 'paragraph=True' can help combine text, but for ID fields, False is better.
            ocr_result = ocr_reader.readtext(cropped_image, detail=0, paragraph=False)
            
            if ocr_result:
                # Join multiple lines if OCR finds them (e.g., for Address)
                full_text = " ".join(ocr_result).strip()
                extracted_data[cls_name] = full_text
            else:
                extracted_data[cls_name] = "" # No text found

    # Draw bounding boxes on the original image for visual debugging
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # You may need to load a font to draw text
    # try:
    #     font = ImageFont.truetype("arial.ttf", 15)
    # except IOError:
    #     font = ImageFont.load_default() # Fallback
        
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cls_name = CLASS_NAMES[class_ids[i]]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        # draw.text((x1, y1 - 10), cls_name, fill="red", font=font)

    return extracted_data, annotated_image

# --- Main Streamlit App UI ---

st.set_page_config(layout="wide", page_title="Customer Onboarding")
st.title("Customer Onboarding Portal")

# Initialize DB
init_db()

# Load models
try:
    yolo_model, ocr_reader = load_models()
except Exception as e:
    st.error(f"Fatal error loading models: {e}")
    st.stop()


# File uploader
uploaded_file = st.file_uploader("Upload an ID (PDF, JPG, PNG)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file:
    # --- 1. Load and Convert File ---
    file_bytes = uploaded_file.getvalue()
    
    if uploaded_file.type == "application/pdf":
        st.write("Converting PDF to image...")
        original_image = pdf_to_image(file_bytes)
    else:
        original_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    if original_image:
        st.write("Processing image...")
        
        # --- 2. Process Image (YOLO + OCR) ---
        with st.spinner("Detecting fields and extracting data..."):
            extracted_data, annotated_image = process_image(original_image, yolo_model, ocr_reader)

        st.subheader("Extraction Results")
        
        # Display annotated image
        st.image(annotated_image, caption="Detected Fields", use_column_width=True)

        if not extracted_data:
            st.error("No data could be extracted. Please try another file.")
        else:
            # --- 3. Display Web Form ---
            st.subheader("Customer Verification Form")
            st.write("Please verify the extracted data and submit.")
            
            with st.form("customer_form"):
                
                # Column for Face
                if "Face" in extracted_data:
                    st.image(extracted_data["Face"], caption="Customer", width=200)
                
                # Dynamically create text inputs for other fields
                form_data = {}
                for field_name in CLASS_NAMES:
                    if field_name != 'Face':
                        # Get the extracted value, default to empty string if not found
                        default_value = extracted_data.get(field_name, "")
                        form_data[field_name] = st.text_input(field_name, value=default_value)
                
                # Submit button
                submitted = st.form_submit_button("Save Customer")

                # --- 4. Save to DB ---
                if submitted:
                    add_customer(form_data)
else:
    st.info("Please upload a file to begin the onboarding process.")

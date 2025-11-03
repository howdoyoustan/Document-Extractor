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

# --- Model Loading (Cached) ---

@st.cache_resource
def load_models():
    """Loads and caches the YOLO and EasyOCR models."""
    print("Loading models...")
    try:
        yolo_model = YOLO(MODEL_PATH)
        ocr_reader = easyocr.Reader(['en']) # Initialize EasyOCR
        print("Models loaded successfully.")
        return yolo_model, ocr_reader
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e

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
    insert_query = f"INSERT INTO customers ({', '.join(keys)}) VALUES ({placeholders})"
    
    try:
        c.execute(insert_query, values)
        conn.commit()
    except Exception as e:
        st.error(f"Database error: {e}")
    finally:
        conn.close()

# --- Image/PDF Processing Functions ---

def pdf_to_image(file_bytes):
    """Converts the first page of a PDF to a PIL Image."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc.load_page(0)  # Load the first page
        pix = page.get_pixmap(dpi=300)  # Render at high resolution
        doc.close()
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    except Exception as e:
        st.error(f"Error converting PDF: {e}")
        return None

def process_image(image, yolo_model, ocr_reader):
    """
    Runs YOLO detection and OCR on the image.
    Returns a dictionary of extracted data and the annotated image.
    """
    # --- 1. YOLOv8 Detection ---
    results = yolo_model.predict(image, conf=0.4) # Use a confidence threshold
    
    extracted_data = {}
    
    # Convert PIL Image to NumPy array for cropping (OpenCV format)
    image_np = np.array(image.convert("RGB"))
    
    # Get the bounding boxes, class IDs, and class names
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    
    if len(boxes) == 0:
        st.warning("YOLO model did not detect any fields. Try a clearer image or adjust model confidence.")
        return {}, image # Return original image if no detections

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
    
    # Fallback font
    font = ImageFont.load_default()
        
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cls_name = CLASS_NAMES[class_ids[i]]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), cls_name, fill="red", font=font)

    return extracted_data, annotated_image

# --- Streamlit Form Display Function ---

def process_and_display(class_names):
    """Displays the Streamlit form using data from session_state."""
    try:
        if "extracted_data" in st.session_state:
            extracted_data = st.session_state.extracted_data
            face_image = st.session_state.face_image

            with st.form("customer_form"):
                st.subheader("Extracted Customer Information")
                st.write("Please verify the data below.")

                if face_image:
                    # CHANGE 2: Display Full Name under face
                    first_name = extracted_data.get("First name", "")
                    last_name = extracted_data.get("Last name", "")
                    full_name = f"{first_name} {last_name}".strip()
                    
                    caption_text = full_name if full_name else "Customer Face"
                    
                    st.image(face_image, caption=caption_text, width=150)
                
                # This dict will hold the FINAL values from the form
                form_data = {}

                # Dynamically create text inputs for all fields except 'Face'
                for field_name in class_names:
                    if field_name == "Face":
                        continue
                    
                    default_value = extracted_data.get(field_name, "")
                    
                    # CHANGE 1: Don't display 'Class' if it's empty
                    if field_name == "Class" and not default_value:
                        form_data[field_name] = "" # Assign the empty string directly
                        continue # Skip creating the st.text_input
                    
                    # Create the input and store its current value
                    form_data[field_name] = st.text_input(f"{field_name}", value=default_value)

                submitted = st.form_submit_button("Save Customer")
                if submitted:
                    # On submit, 'form_data' now contains all the edited values
                    add_customer(form_data) # Corrected function call
                    st.success("Customer saved successfully!")
                    
                    # Clear session state to reset the app
                    keys_to_clear = ["extracted_data", "face_image", "file_processed", "annotated_image"]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun() # Rerun to show the file uploader again

    except Exception as e:
        st.error(f"Error displaying form: {e}")

# --- Main Streamlit App UI ---

st.set_page_config(layout="wide", page_title="Customer Onboarding")
st.title("Customer Onboarding Portal")

# Initialize DB
init_db()

# Load models
try:
    yolo_model, ocr_reader = load_models()
except Exception as e:
    st.error(f"Fatal error loading models: {e}. Please check model path and dependencies.")
    st.stop()


# File uploader
uploaded_file = st.file_uploader("Upload an ID (PDF, JPG, PNG)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file:
    # Use session state to avoid reprocessing on form submission
    if "file_processed" not in st.session_state:
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
            
            if not extracted_data:
                st.error("No data could be extracted. Please try another file.")
                # Clear state and stop
                keys_to_clear = ["extracted_data", "face_image", "file_processed", "annotated_image"]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.stop()
            
            # Store results in session state
            st.session_state.extracted_data = extracted_data
            st.session_state.annotated_image = annotated_image
            st.session_state.face_image = extracted_data.get("Face", None)
            st.session_state.file_processed = True
        else:
            st.error("Could not load image.")
            st.stop()

    # --- 3. Display Results and Form ---
    # This part runs every time, reading from session state
    if "file_processed" in st.session_state:
        st.subheader("Extraction Results")
        st.image(st.session_state.annotated_image, caption="Detected Fields", use_column_width=True)
        
        # Call the function that displays the form
        process_and_display(CLASS_NAMES)

else:
    # Clear session state if file is removed or app is reset
    keys_to_clear = ["extracted_data", "face_image", "file_processed", "annotated_image"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.info("Please upload a file to begin the onboarding process.")


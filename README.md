Automated Customer Onboarding with ID Verification

Welcome to the Automated Customer Onboarding portal, a sophisticated solution designed to streamline and accelerate the new customer journey. This application leverages cutting-edge AI to transform a tedious manual data-entry task into a simple, fast, and accurate automated process.

🚀 Overview

This project's goal is to revolutionize customer onboarding. By simply uploading an ID document (such as a driver's license in PDF or image format), this tool intelligently handles the rest. It scans the document, identifies key information, extracts the text, and pre-populates a digital form.

This eliminates human error, drastically reduces onboarding time, and provides a seamless experience for the end-user. The verified data is then securely stored in a local database for record-keeping.

🛠️ Technology Stack

This application is built with a modern, powerful, and efficient tech stack:

User Interface: Streamlit

Object Detection: YOLOv8 (Ultralytics)

Text Recognition (OCR): EasyOCR

PDF Handling: PyMuPDF (fitz)

Database: SQLite3

Image Processing: Pillow & OpenCV

✨ Key Features

Flexible File Handling: Accepts standard document formats, including PDF, PNG, and JPG/JPEG.

Smart PDF Conversion: Automatically converts uploaded PDFs into high-resolution images for analysis.

AI-Powered Detection: A custom-trained YOLOv8 model instantly locates 10 distinct information fields on the ID.

Accurate Text Extraction: Employs EasyOCR to read and digitize the text from the identified fields.

Automatic Face Display: Intelligently detects and crops the customer's face from the ID for visual verification.

Dynamic Web Form: Generates an editable form pre-filled with all extracted data, allowing for easy review and confirmation.

Secure Data Persistence: Saves the final, verified customer information to a local SQLite database (customers.db).

⚙️ Setup & Installation

To get the application running on a local machine, follow these two simple steps:

Clone/Download: Place the project files into a dedicated folder (your_app_folder/).

Install Dependencies: Open a terminal in the project folder and install the required packages.

pip install -r requirements.txt


🏁 How to Use

Running the project involves two phases: a one-time training step and then running the application.

Step 1: Train the YOLOv8 Model (Run Once)

Ensure the data.yaml file and the dataset folders (train, valid, test) are correctly placed as per the original project guidelines.

Run the training script from the project folder:

python train.py


After training, a new runs/ folder will be created. Navigate to runs/detect/id_card_detector/weights/ and find the best.pt file.

Copy this best.pt file into the main your_app_folder/.

Rename the copied file to best_model.pt.

Step 2: Run the Streamlit Application

With the best_model.pt file in place, start the Streamlit app:

streamlit run app.py


Your web browser will automatically open to the application's local URL, ready for use.

📊 Dataset Classes

The AI model was trained to identify 10 specific classes of information:

Address

Class

DOB (Date of Birth)

Exp date (Expiration Date)

Face

First name

Issue date

Last name

License number

Sex

# **Automated Customer Onboarding with ID Verification**

Welcome to the Automated Customer Onboarding portal, a sophisticated solution designed to streamline and accelerate the new customer journey. This application leverages cutting-edge AI to transform a tedious manual data-entry task into a simple, fast, and accurate automated process.

## **🚀 Overview**

This project's goal is to revolutionize customer onboarding. By simply uploading an ID document (such as a driver's license in PDF or image format), this tool intelligently handles the rest. It scans the document, identifies key information, extracts the text, and pre-populates a digital form.

This eliminates human error, drastically reduces onboarding time, and provides a seamless experience for the end-user. The verified data is then securely stored in a local database for record-keeping.

## **🛠️ Technology Stack**

This application is built with a modern, powerful, and efficient tech stack:

* **User Interface:** [Streamlit](https://streamlit.io/)  
* **Object Detection:** [YOLOv8 (Ultralytics)](https://www.google.com/search?q=https://ultralytics.com/)  
* **Text Recognition (OCR):** [EasyOCR](https://github.com/JaidedAI/EasyOCR)  
* **PDF Handling:** [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/en/latest/)  
* **Database:** [SQLite3](https://www.sqlite.org/index.html)  
* **Image Processing:** Pillow & OpenCV

## **✨ Key Features**

* **Flexible File Handling:** Accepts standard document formats, including PDF, PNG, and JPG/JPEG.  
* **Smart PDF Conversion:** Automatically converts uploaded PDFs into high-resolution images for analysis.  
* **AI-Powered Detection:** A custom-trained YOLOv8 model instantly locates 10 distinct information fields on the ID.  
* **Accurate Text Extraction:** Employs EasyOCR to read and digitize the text from the identified fields.  
* **Automatic Face Display:** Intelligently detects and crops the customer's face from the ID for visual verification.  
* **Dynamic Web Form:** Generates an editable form pre-filled with all extracted data, allowing for easy review and confirmation.  
* **Secure Data Persistence:** Saves the final, verified customer information to a local SQLite database (customers.db).

## **⚙️ Setup & Installation**

To get the application running on a local machine, follow these two simple steps:

1. **Clone/Download:** Place the project files into a dedicated folder (your\_app\_folder/).  
2. **Install Dependencies:** Open a terminal in the project folder and install the required packages.  
   pip install \-r requirements.txt

## **🏁 How to Use**

Running the project involves two phases: a one-time training step and then running the application.

### **Step 1: Train the YOLOv8 Model (Run Once)**

1. Ensure the data.yaml file and the dataset folders (train, valid, test) are correctly placed as per the original project guidelines.  
2. Run the training script from the project folder:  
   python train.py

3. After training, a new runs/ folder will be created. Navigate to runs/detect/id\_card\_detector/weights/ and find the best.pt file.  
4. **Copy** this best.pt file into the main your\_app\_folder/.  
5. **Rename** the copied file to best\_model.pt.

### **Step 2: Run the Streamlit Application**

1. With the best\_model.pt file in place, start the Streamlit app:  
   streamlit run app.py

2. Your web browser will automatically open to the application's local URL, ready for use.

## **📊 Dataset Classes**

The AI model was trained to identify 10 specific classes of information:

1. Address  
2. Class  
3. DOB (Date of Birth)  
4. Exp date (Expiration Date)  
5. Face  
6. First name  
7. Issue date  
8. Last name  
9. License number  
10. Sex
# **YOLO-based Customer Onboarding App**

This project provides a complete, production-ready starter application for customer onboarding. It allows users to upload an ID document (PDF or image), extracts key information and the customer's face using a YOLOv8 model, and saves that data to a SQLite database.

## **Project Structure**

/  
├── app.py              \# Main Flask web server  
├── yolo\_helpers.py     \# Functions for YOLO training & inference  
├── database.py         \# Script to initialize the SQLite DB  
├── requirements.txt    \# Python dependencies  
|  
├── /static/  
│   ├── /uploads/       \# Original user uploads are saved here  
│   └── /faces/         \# Cropped face images are saved here  
|  
├── /templates/  
│   ├── index.html      \# The file upload page  
│   └── form.html       \# The generated web form page  
|  
├── /dataset/           \# \--- YOU MUST CREATE THIS \---  
│   ├── /images/  
│   │   ├── train/  
│   │   └── val/  
│   ├── /labels/  
│   │   ├── train/  
│   │   └── val/  
│   └── data.yaml       \# YOLO dataset configuration file  
|  
└── /runs/              \# YOLO training/inference results (auto-generated)  
    └── /detect/  
        └── /train/  
            └── /weights/  
                └── best.pt \# Your final trained model\!

## **Step 1: Pre-training Your YOLO Model (Crucial\!)**

You cannot use a generic YOLO model for this. You **must** train it to recognize the specific fields on *your* ID documents.

### **1.1. Data Annotation**

1. **Collect Data:** Gather 100+ sample images of the ID documents you want to process.  
2. **Annotate:** Use an annotation tool like **Roboflow** or **CVAT**. You must draw bounding boxes around each piece of information you want to extract.  
   * Create a **class** for each field: e.g., name, dob, id\_number, face.  
3. **Export:** Export your annotations in **YOLOv8 format**. This will give you a dataset folder containing images, labels, and a data.yaml file. Place this dataset folder in your project directory as shown in the structure above.

### **1.2. Edit data.yaml**

Your data.yaml file (from your annotation tool) should look something like this. Make sure the paths are correct.

\# dataset/data.yaml  
train: ../dataset/images/train  
val: ../dataset/images/val

\# number of classes  
nc: 4

\# class names  
names: \['name', 'dob', 'id\_number', 'face'\]

### **1.3. Run Training**

Before you can run the web app, you must train your model.

1. Install the dependencies:  
   pip install \-r requirements.txt

2. Run the training script from yolo\_helpers.py:  
   python \-c "from yolo\_helpers import train\_model; train\_model()"

3. This process will take time (minutes to hours depending on your data and GPU). When finished, it will create a file at runs/detect/train/weights/best.pt. **This is your trained model.**

## **Step 2: Set Up & Run the Web Application**

### **2.1. Initialize the Database**

Run the database.py script **one time** to create your onboarding.db file and the customers table.

python database.py

### **2.2. Run the Flask App**

Now you can start the main server.

flask \--app app \--debug run

This will start the server, typically at http://127.0.0.1:5000.

## **How It Works**

1. Go to http://127.0.0.1:5000 in your browser.  
2. Upload a PDF or image of an ID card.  
3. **Backend Process:**  
   * The file is saved to /static/uploads/.  
   * If it's a **PDF**, app.py uses **PyMuPDF** to convert the first page to a high-resolution PNG.  
   * The run\_inference function from yolo\_helpers.py is called.  
   * Your trained model (best.pt) analyzes the image.  
   * It finds all the data fields and their bounding boxes.  
   * The face is cropped and saved to /static/faces/.  
   * The extracted text fields (name, DOB, etc.) are saved to the **SQLite** database.  
4. You are redirected to the /form/\<id\> page, which displays the data and face you just extracted.
# DS Final Project Documentation: Number Plate & Vehicle Detection

## Project Overview
This project implements an end-to-end Automatic Number Plate Recognition (ANPR) system. The pipeline combines YOLO-based object detection with OCR (Optical Character Recognition) to detect number plates, extract them, and convert them into text. The solution is designed for scalability and deployment in a cloud-based web application. 

## Step 1: Install Dependencies
The first step in building the project pipeline was setting up the working environment with all necessary dependencies. Since Google Colab was used as the primary development platform, I leveraged its GPU acceleration and pre-installed deep learning libraries. The main packages required were YOLOv8 for object detection, Torch for model training, OpenCV for image processing, and Tesseract/EasyOCR for optical character recognition. Additionally, auxiliary packages like shutil, os, random, and pandas were installed for file handling and structured dataset preparation. Ensuring the correct versions of TensorFlow/PyTorch and other libraries was essential to maintain compatibility during training and deployment phases.

## Step 2: Mount drive and unzip files
### 2.1 Mount the drive with dataset:
Since I am using colab, if I upload the files directly, it takes forever to load them everytime, if runtime disconnecst from server, hence mounting the google drive with labelled dataset directly. The dataset was stored in Google Drive and mounted in Colab using the google.colab drive API. This allowed seamless access to large image and label files without manual upload limitations. After mounting, the dataset was extracted from compressed .zip files using the unzip utility. Directory structures were created with mkdir to organize the dataset properly before preprocessing. At this stage, the dataset contained both raw images and annotation text files (labels), which were required to train YOLO for number plate detection.

## Step 3: #Splitting data for the correct YOLO Structure
YOLO requires a strict dataset organization into train, validation, and test splits, with separate folders for images and labels. We wrote a custom Python script to automate this splitting process. First, all available image files and corresponding label files were identified. A shuffle operation was applied to ensure randomness, followed by an 80-10-10 ratio split for training, validation, and testing respectively. 

## Step 4: Creating data.yaml
YOLO requires a configuration file in YAML format to specify dataset details. I created a data.yaml file that included the relative paths to the train, validation, and test directories along with the class definitions. Since the focus was on vehicle number plate detection, only a single class was defined.

## Step 5: Data Preprocessing
Preprocessing was an important step to enhance data quality and ensure robust training. Images were resized to a consistent input dimension (commonly 640x640 for YOLO) to maintain uniformity across batches. Pixel values were normalized to a [0,1] range, which accelerated convergence during training. Data augmentation techniques such as random rotation, brightness and contrast adjustment, horizontal flipping, and Gaussian noise injection were applied to increase dataset diversity. Additionally, low-quality images with excessive blur or noise were filtered out, ensuring only clean samples were used for training. This step significantly improved the model’s ability to generalize to real-world test cases.
YOLOv8 automatically resizes input images to the images and normalizes pixel values internally. So manual resizing is optional, but sometimes to standardize resolution to save GPU memory, we can manually resize

## Step 6: Model Training with YOLO
The YOLO (You Only Look Once) object detection framework is used due to its efficiency in real-time detection tasks. I trained the model using GPU acceleration available in Colab. Hyperparameters such as learning rate, batch size, and number of epochs were fine-tuned for optimal performance. During training, YOLO learned to localize number plates within images by predicting bounding boxes and class probabilities. The training progress was monitored using real-time loss curves, precision, recall, and mean Average Precision (mAP) metrics. After sufficient epochs, the best-performing model weights were saved for inference.

## Step 7: Post-processing and OCR extraction
Once YOLO successfully detected number plates, the next step was extracting textual information. Detected bounding box coordinates were used to crop out the number plate region from each image. These cropped images were preprocessed further using binarization, contrast enhancement, and noise reduction techniques to improve text visibility. The processed plate regions were then passed to OCR engines (Tesseract and EasyOCR), which converted the image regions into machine-readable text. Extracted results were saved in a structured CSV file containing the image name, detected bounding box, and recognized plate text. This created a bridge between raw detection and usable alphanumeric license plate information.

## Step 8: Streamlit app development and deployment
### 8.1 App development:
The Streamlit-based ANPR (Automatic Number Plate Recognition) application was developed to provide a user-friendly interface for real-time vehicle and number plate detection. The app was built in Python using the streamlit library for the web interface, ultralytics YOLO framework for object detection, and OpenCV along with EasyOCR for image processing and text extraction. The application allows users to upload images in supported formats such as PNG, JPG, JPEG, and HEIC. Upon uploading, the YOLO model detects vehicles and number plates in the image, crops the detected plates, and passes them to the OCR pipeline. Preprocessing steps including image binarization and contrast enhancement were applied to improve OCR accuracy. Detected plates and their corresponding text were then displayed in a structured format within the Streamlit interface.
The project structure includes app.py as the main Streamlit script, a trained YOLO model (best.pt), and necessary Python dependencies listed in a virtual environment. The app ensures compatibility by checking OpenCV version and verifying the presence of required shared libraries such as libGL.so.1.

### 8.2 AWS EC2 Instance Configuration
For deployment, an AWS EC2 instance was launched to host the Streamlit application. An Ubuntu 24.04 LTS AMI was chosen with a t2.medium instance type, providing sufficient CPU and memory resources for inference. During instance creation, a security group was configured to allow inbound traffic on SSH (port 22) and HTTP/Streamlit (port 8501) from anywhere, enabling both remote terminal access and web access. A key pair (ANPD.pem) was used to securely connect to the instance via SSH.
After launching, the EC2 instance was accessed from a Windows terminal 

This provided a secure remote shell for installing dependencies, setting up virtual environments, and transferring project files.

### 8.3 Project Setup on EC2
Once connected, a Python virtual environment (anpr-env) was created to isolate dependencies:
python3 -m venv anpr-env
source anpr-env/bin/activate
Dependencies including streamlit, ultralytics, opencv-python-headless, easyocr, and other libraries were installed within this environment using pip. The project files app.py and best.pt were uploaded from the local machine to the EC2 instance using scp:
Directory navigation and verification were performed to ensure all files were correctly placed in the home directory.

### 8.4 Running the Streamlit App:
With the environment activated and files in place, the Streamlit app was launched on the EC2 instance using:
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
This command explicitly binds the server to all network interfaces (0.0.0.0) and port 8501, allowing access from a browser on any machine using the EC2 public IP address:

Once running, the web application allowed uploading images from the browser. The YOLO model processed the uploaded images, detected number plates, and the OCR pipeline extracted plate text, displaying both the cropped plates and recognized numbers on the interface.
## Final Flow in the App:
1.	You upload a vehicle image.
2.	Streamlit saves it temporarily.
3.	YOLO model predicts plate bounding box.
4.	The cropped plate region is processed by OCR.
5.	Streamlit displays:
o	Original image with detection box
o	Extracted plate text

## Summary
In summary, the deployment workflow involves preparing a Streamlit-based ANPR application, launching and configuring an EC2 instance, setting up a Python environment, uploading project files, installing dependencies, and running the application in a publicly accessible manner. The project ensures robust detection and text extraction by combining YOLO object detection with OCR preprocessing, all accessible through a simple web interface hosted on AWS EC2.

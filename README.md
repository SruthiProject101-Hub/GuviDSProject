## 1. Project Overview:
This project aims to develop a deep learning-based solution for classifying chest X-ray images into either normal or showing signs of Tuberculosis (TB). The primary goal was to build a model capable of assisting clinicians in early detection of TB using automated image classification. The solution leverages transfer learning, using pre-trained convolutional neural networks such as VGG16, ResNet50, and EfficientNetB0, with custom dense layers for binary classification. To make the solution accessible, a web application was developed using Streamlit and deployed on an AWS EC2 instance.

## 2. Environment Setup:
The development environment was carefully configured to ensure compatibility between TensorFlow, Keras, and other required packages. For local development, Python 3.9 was used along with TensorFlow 2.18.0, Pillow for image processing, numpy for numerical computations, and Streamlit for web interface development. On the AWS EC2 instance, Ubuntu 24.04 served as the operating system, and a dedicated virtual environment was created to isolate dependencies and avoid version conflicts. Using a virtual environment also ensured reproducibility and easier deployment. All necessary Python packages were installed via pip, ensuring that TensorFlow and Keras were compatible for model loading and prediction.

## 3. Data Loading and Exploration:
The dataset consisted of chest X-ray images, divided into two categories: TB-affected and normal. In total, 3008 images were available, with 2494 showing TB and 514 being normal. The images were organized into separate folders, which allowed for systematic data loading. Using Python libraries such as os and cv2, the images were read and counted to verify the dataset distribution. This step revealed the significant class imbalance, which required careful handling during model training. Initial exploration also included checking image sizes and formats to ensure consistency.

## 4. Data Preprocessing:
Preprocessing of images was an essential step to prepare the data for model training. All images were resized to 224x224 pixels to match the input requirements of the pre-trained CNN models. Images were converted to RGB format to ensure three color channels, even if originally grayscale, and pixel values were normalized to the range [0,1]. Data augmentation techniques such as rotations, flipping, zooming, and shifts were applied to improve model generalization. Additionally, due to class imbalance, strategies like assigning class weights during training were implemented to prevent the model from being biased towards the TB class.

## 5. Exploratory Data Analysis (EDA):
Exploratory data analysis provided insights into the distribution and characteristics of the dataset. Visualization of sample images allowed verification that both TB and normal images were correctly labeled. Histograms of pixel intensities helped understand image quality and lighting conditions. A bar chart illustrating class distribution highlighted the imbalance between TB and normal images, confirming the need for class weighting during training. This step ensured that the model developer had a clear understanding of the dataset before proceeding to model building
By the end of EDA, we figured out 1. The dataset is imbalanced (TB >> Normal), 2. Pixel intensity ranges and contrast levels 3. If resizing is required (for CNNs) and 4. Visual differences between classes.

## 6.Model Development:
Transfer learning was employed to build the classification model efficiently. Pre-trained models such as VGG16, ResNet50, and EfficientNetB0 were chosen due to their proven performance in image recognition tasks. The top layers of these networks were replaced with custom dense layers suitable for binary classification, including a flattening layer, a fully connected layer with ReLU activation, a dropout layer for regularization, and a final sigmoid layer for output probability. The model was compiled using the Adam optimizer with a binary cross-entropy loss function. By fine-tuning the top layers of the network, the model learned to identify features relevant to TB detection without requiring extensive training from scratch.

## 7. Model Evaluation and comparison:
1. VGG16 outperforms the others on accuracy (92%) and F1-Score (95%), meaning it balances precision and recall better.
2. ResNet50 and EfficientNetB0 both have perfect recall (1.0), meaning they caught all TB cases (no false negatives). This is crucial in medical diagnosis.
•	But their precision is lower (83%), which means there are more false positives (some normal X-rays wrongly classified as TB).
3. VGG16’s recall (98%) is almost perfect and precision is higher (92%), so it’s better at avoiding false alarms while still catching nearly all TB cases.
Therefore, VGG16 is the best model among the three.

## 8.Streamlit App Development:
A user-friendly web application was developed using Streamlit to deploy the model for practical use. The app allows users to upload chest X-ray images, preprocesses them to the required format, and predicts whether the image shows TB. To improve performance, the model is loaded once at startup and cached to avoid repeated loading. The interface displays the uploaded image, prediction results, and confidence scores, providing an intuitive experience for medical practitioners or other users. Error handling is included to manage unsupported file formats or corrupted images.

## 9. Local testing:
Before deployment, the application was tested locally to ensure functionality. Using the streamlit run app.py command, the app could be launched in a browser. Different test images, including normal and TB X-rays, were uploaded to verify that the model predictions were accurate.

## 10. Deployment to AWS:
To make the application accessible over the web, an AWS EC2 Ubuntu instance was launched. Python and necessary dependencies were installed within a virtual environment. The application files, including the saved model, were securely copied to the instance using scp. Security group rules were configured to allow inbound TCP traffic on port 8501, the default Streamlit port. Finally, the Streamlit app was launched on the server, binding to 0.0.0.0 to accept connections from all IP addresses. The application became accessible via the EC2 public IP address, allowing users to interact with it remotely.



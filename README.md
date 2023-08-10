# YOLO_CNN_Helmet_Detection_With_Flask
Welcome to the YOLO-CNN Helmet Detection project! This project introduces a new and practical method for helmet detection in industrial environments. Traditional computer vision methods often struggle with accuracy, and while artificial intelligence approaches have potential, real-time performance remains a challenge. This project aims to bridge this gap by combining YOLO-based person detection with a custom helmet detection method, all optimized for real-world scenarios.

Method Overview
YOLO Person Detection: We start by utilizing the YOLO (You Only Look Once) object detection method to identify individuals within frames. YOLO's efficiency and accuracy make it an excellent choice for this initial step.

Helmet Detection: After person detection, we apply a specialized helmet detection method to determine whether a detected person is wearing a helmet or not. This two-stage approach enhances accuracy and contextual understanding.

Frame Skipping Optimization: To improve processing speed without sacrificing effectiveness, we employ frame skipping. This means we process every nth frame, leveraging the assumption of minimal changes between consecutive frames in many scenarios.

Dataset
The dataset used for this project is custom-built, tailored to industrial settings. To fine-tune the models, we encourage using your own images, which align more closely with the specific conditions you aim to address. The dataset personalization ensures the best performance in your application environment.

Getting Started
Use the create_cnn_weight.py script provided to create your own CNN weight. By using your custom images, you can fine-tune the CNN model to suit your application's needs.

Implement the two-stage YOLO-CNN method in your application, following the guidelines provided in the app.py script.

Usage
To apply this YOLO-CNN Helmet Detection method to your own scenarios, follow these steps:

Prepare your custom dataset with images from your industrial environment.

Fine-tune the CNN model using the create_cnn_weight.py script and your custom images.

Integrate the trained models and the two-stage process into your application, enhancing helmet detection accuracy in real-time situations.

Feel free to adapt and extend this project according to your needs, contributing to improved safety and efficiency in industrial workplaces.


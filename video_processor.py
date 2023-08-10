# Import necessary libray
from ultralytics import YOLO
import numpy as np
import cv2
import cvzone
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
from PIL import Image


# Define the videoProcessor class
class VideoProcessor:
    """
    Initializes the object by defining parameters and loading necessary weight for futher processing
    Sets confidence thershold for helmet detection, video path , frame processing parameters,
    and image enhancement parameters
    """

    def __init__(self, video_path):
        # Initialize YOLO for person detection
        self.person_detector = YOLO(r'cocoL.pt')

        # Load the CNN weight for helmet detection to identify helmets in images
        self.helmet_detector = load_model(r'final_cnn_helmet_detection_weight.h5')

        # Confidence thresholds for helmet detection

        # If the prediction confidence is higher than 0.65, we can consider it reliable; otherwise,
        # we will ignore it
        self.no_helmet_confidence_threshold = 0.65
        # If the prediction confidence is lower than 0.35, we can consider it reliable ; otherwise,
        # we will ignore it
        self.helmet_confidence_threshold = 0.35

        # Path to the input video
        self.video_path = video_path

        # Parameters for skipping frames during processing.
        self.frame_skip_interval = 5
        self.frame_count = 0

        # Image upscaling parameters
        self.scale_factor = 2
        self.iterations = 2

        # Image size
        self.image_size = (224, 224)

    """    
    this function enhances the resolutin of the input image by iteratively applying upscaling .
    the image is resized using a specified scale factor , and the process is repeated a set number
    of times to gradually improve the resolution. This helps compensate for varying image resolutions
    across different environments
    """

    def upscale_image(self, image_array):

        # Convert the image array to PIL Image
        img = Image.fromarray(np.uint8(image_array))

        # Perform image upscaling based on the specified number of iterations
        for i in range(self.iterations):
            width, height = img.size
            new_width = width * self.scale_factor
            new_height = height * self.scale_factor

            # Conver the upscaled image to numpy array
            img = img.resize((new_width, new_height), Image.LANCZOS)
        upscaled_image_array = np.array(img)
        return upscaled_image_array

    """
    This function focuses on the head region of the detected person by cropping
    the specified area around the bounding box. It then utilizes the 'upscale_image'
    function to improve the resolution of the cropped image. Finally, the image is
    converted to an array suitable for further processing by the CNN helmet detection model.
    """

    def prepare_cropped_image(self, img, x1, y1, x2, y2):

        # Calculate the dimensions for cropping  to isolate the head in the image.
        y3 = y2 - y1
        y5 = int(y3 * 0.3 + y1)
        y4 = int(y1 - y3 * 0.05)
        crop = img[y4:y5, x1:x2]

        # Upscale the cropped image
        improve = self.upscale_image(crop)

        # Conver upscaled image to a numpy array and resize it the appropriate  dimensions
        # for input to the prediction model
        resized_img = array_to_img(improve, scale=False).resize(self.image_size)
        img_array = img_to_array(resized_img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        return img_array

    """
    This function annotates the input image by drawing bounding boxes around detected persons.
    Depending on the helmet detection prediction, the bounding box is colored red for no helmet
    and green for helmet. Labels indicating 'no helmet' or 'helmet' are placed above the respective
    bounding boxes. The annotation process enhances the visual representation of helmet detection
    results on the image 
    """

    def annotate_image(self, img, prediction, x1, y1, x2, y2):
        # if the confidence of the prediction for detection a helmet is greater than 0.65 continue
        # with the processing
        if prediction[0][0] > self.no_helmet_confidence_threshold:

            # Draw red rectangle and label for "no helmet"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cvzone.putTextRect(img, 'no helmet', (max(0, x1), max(35, y1)),
                               scale=0.8, thickness=1, offset=10)

        # if the confidence of the prediction for detection a helmet is less than 0.35 continue
        # with the processing
        elif prediction[0][0] < self.helmet_confidence_threshold:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cvzone.putTextRect(img, 'helmet', (max(0, x1), max(35, y1)),
                               scale=0.8, thickness=1, offset=10)

    """
    Processes the input video by detecting persons and their helmet status.
    This function reads the input video and processes each frame. It employs frame skipping
    to process one frame out of every five frames. This optimization helps enhance real-time
    processing speed while maintaining accuracy, as the environment might not have significant
    changes between frames. Within each processed frame, persons are detected using YOLO, and
    their helmet status is determined using the helmet detection model. Bounding boxes are drawn
    around the detected persons, and labels are added to indicate helmet or no helmet. The processed
    results are displayed in the video, providing visual feedback on helmet detection     
    """

    def process_video(self):

        # Open the video file
        cap = cv2.VideoCapture(self.video_path)

        # Read a frame from the video
        while True:
            success, img = cap.read()
            if not success:
                break  # No more frames to process, exit the loop

            # Increment the frame count by one for tracking the number processed frames
            self.frame_count += 1

            # Skip processing the current frame if its count is not a multiple of the frame skip interval
            if self.frame_count % self.frame_skip_interval != 0:
                continue

            results = self.person_detector(img, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    if box.cls == 0:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        img_array = self.prepare_cropped_image(img, x1, y1, x2, y2)
                        prediction = self.helmet_detector.predict(img_array)
                        self.annotate_image(img, prediction, x1, y1, x2, y2)
            # Display the annotated image
            cv2.imshow('image', img)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
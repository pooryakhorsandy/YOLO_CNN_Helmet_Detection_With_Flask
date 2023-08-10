# Import necessary libraries
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from video_processor import VideoProcessor
import os

# Create a Flask app instance
app = Flask(__name__)

# Define the route for the main page
@app.route('/')
def index():
    return render_template('upload.html')

# Define the route for processing the uploaded video
@app.route('/process_video', methods=['POST'])
def process_video():
    # Get the uploaded video file from the form
    video_file = request.files['video']
    if video_file:
        # Construct the path to save the uploaded video
        video_path = os.path.join('your image folder path', secure_filename(video_file.filename))
        # Save the uploaded video to the specified path
        video_file.save(video_path)

        # Initialize the video processor with the uploaded video
        video_processor = VideoProcessor(video_path)

        # Process the video using the video processor
        video_processor.process_video()

    # Redirect back to the main page after processing
    return redirect(url_for('index'))

# Run the Flask app if this script is executed directly
if __name__ == '__main__':
    app.run(debug=True)

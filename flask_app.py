from flask import Flask, render_template, jsonify, send_from_directory
import os

from shared_data import ocr_results  # Import the shared variable

# Flask app setup
app = Flask(__name__)

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/results')
def get_results():
    """Return the latest OCR results as JSON."""
    return jsonify(ocr_results)

@app.route('/cropped_images/<filename>')
def serve_cropped_image(filename):
    """Serve cropped images."""
    return send_from_directory("cropped_images", filename)

@app.route('/still_shots/<filename>')
def serve_still_shot(filename):
    """Serve still shot images."""
    return send_from_directory("still_shots", filename)

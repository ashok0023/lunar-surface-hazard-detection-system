#!/usr/bin/env python3
"""
Lunar Hazard Detection Web Application
Beautiful and colorful web interface for lunar surface hazard detection.
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'web_app/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

# Create upload directory
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# Lunar hazard class colors and information
LUNAR_CLASSES = {
    0: {'name': 'Crater', 'color': '#FF6B6B', 'description': 'Impact craters of various sizes'},
    1: {'name': 'Rock', 'color': '#4ECDC4', 'description': 'Surface rocks and boulders'},
    2: {'name': 'Shadow Region', 'color': '#45B7D1', 'description': 'Permanently shadowed areas'},
    3: {'name': 'Dust Devil', 'color': '#96CEB4', 'description': 'Dust movement patterns'},
    4: {'name': 'Slope', 'color': '#FFEAA7', 'description': 'Steep terrain features'},
    5: {'name': 'Lunar Module', 'color': '#DDA0DD', 'description': 'Human-made objects'}
}

# Global model variable
model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    """Load the YOLO model"""
    global model
    try:
        # Try to load our trained lunar model first
        trained_model_path = 'runs/detect/yolov8_lunar_20250923_013613/weights/best.pt'

        if os.path.exists(trained_model_path):
            model_path = trained_model_path
            logger.info(f"Loading trained lunar model: {model_path}")
        else:
            model_path = 'yolov8n.pt'  # Fallback to pre-trained model
            logger.info(f"Trained model not found, using pre-trained model: {model_path}")

        model = YOLO(model_path)
        logger.info(f"Model loaded successfully: {model_path}")
        logger.info(f"Model type: {type(model)}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def detect_hazards(image_path: str) -> Dict:
    """Detect hazards in lunar surface image"""
    try:
        # Run inference
        results = model(image_path)

        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    detection = {
                        'class': int(box.cls[0]),
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist(),
                        'class_name': LUNAR_CLASSES[int(box.cls[0])]['name'],
                        'color': LUNAR_CLASSES[int(box.cls[0])]['color']
                    }
                    detections.append(detection)

        return {
            'success': True,
            'detections': detections,
            'total_detections': len(detections),
            'image_path': image_path
        }

    except Exception as e:
        logger.error(f"Error during detection: {e}")
        return {
            'success': False,
            'error': str(e),
            'detections': [],
            'total_detections': 0
        }

def draw_detections(image_path: str, detections: List[Dict]) -> str:
    """Draw bounding boxes on image and return base64 encoded result"""
    try:
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            color = detection['color'].lstrip('#')
            color_rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))

            # Draw bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color_rgb, 3)

            # Draw label
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            cv2.putText(image, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_rgb, 2)

        # Convert to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return img_base64

    except Exception as e:
        logger.error(f"Error drawing detections: {e}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html',
                         lunar_classes=LUNAR_CLASSES,
                         title="Lunar Hazard Detection System")

@app.route('/detect', methods=['POST'])
def detect():
    """Handle image upload and detection"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run detection
        result = detect_hazards(filepath)

        if result['success']:
            # Draw detections on image
            result_image = draw_detections(filepath, result['detections'])

            # Prepare response
            response = {
                'success': True,
                'detections': result['detections'],
                'total_detections': result['total_detections'],
                'result_image': result_image,
                'image_path': filepath
            }

            # Add detection summary
            class_counts = {}
            for detection in result['detections']:
                class_name = detection['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            response['class_summary'] = class_counts

            return jsonify(response)
        else:
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Error in detect route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/sample-images')
def get_sample_images():
    """Get list of sample lunar images"""
    sample_dir = Path('data/augmented')
    if sample_dir.exists():
        images = [f.name for f in sample_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        return jsonify({'images': images[:12]})  # Return first 12 images
    return jsonify({'images': []})

@app.route('/data/augmented/<filename>')
def serve_sample_image(filename):
    """Serve sample images"""
    try:
        sample_dir = Path('data/augmented')
        image_path = sample_dir / filename

        if image_path.exists():
            return send_file(str(image_path), mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/sample-detection/<filename>')
def sample_detection(filename):
    """Run detection on sample image"""
    try:
        sample_path = Path('data/augmented') / filename
        if not sample_path.exists():
            return jsonify({'error': 'Sample image not found'}), 404

        result = detect_hazards(str(sample_path))
        if result['success']:
            result_image = draw_detections(str(sample_path), result['detections'])

            response = {
                'success': True,
                'detections': result['detections'],
                'total_detections': result['total_detections'],
                'result_image': result_image,
                'image_name': filename
            }

            # Add detection summary
            class_counts = {}
            for detection in result['detections']:
                class_name = detection['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            response['class_summary'] = class_counts

            return jsonify(response)
        else:
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Error in sample detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

# Create templates on import
def create_templates():
    """Create HTML templates"""
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)

    # Main index template
    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --warning-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            --danger-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        }

        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .navbar {
            background: var(--dark-gradient) !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }

        .card {
            border: none;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
        }

        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            background: rgba(255,255,255,0.1);
            transition: all 0.3s ease;
        }

        .upload-area:hover {
            border-color: #764ba2;
            background: rgba(255,255,255,0.2);
        }

        .upload-area.dragover {
            border-color: #4facfe;
            background: rgba(79, 172, 254, 0.1);
        }

        .detection-result {
            background: white;
            border-radius: 15px;
            overflow: hidden;
        }

        .hazard-badge {
            padding: 8px 16px;
            border-radius: 25px;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.8rem;
        }

        .stats-card {
            background: var(--primary-gradient);
            color: white;
            border-radius: 15px;
        }

        .btn-custom {
            border-radius: 25px;
            padding: 12px 30px;
            font-weight: bold;
            text-transform: uppercase;
            transition: all 0.3s ease;
        }

        .btn-custom:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .loading-spinner {
            display: none;
            width: 50px;
            height: 50px;
            border: 5px solid rgba(255,255,255,0.3);
            border-top: 5px solid white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .result-image {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .sample-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .sample-item {
            position: relative;
            overflow: hidden;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.3s ease;
        }

        .sample-item:hover {
            transform: scale(1.05);
        }

        .sample-item img {
            width: 100%;
            height: 150px;
            object-fit: cover;
        }

        .sample-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(transparent, rgba(0,0,0,0.7));
            color: white;
            padding: 10px;
            font-size: 0.8rem;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container">
            <a class="navbar-brand" href="#">
                <i class="fas fa-moon me-2"></i>
                Lunar Hazard Detection System
            </a>
        </div>
    </nav>

    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="card mb-4">
                    <div class="card-body text-center p-5">
                        <h1 class="card-title mb-4">
                            <i class="fas fa-satellite text-primary me-3"></i>
                            Lunar Surface Hazard Detection
                        </h1>
                        <p class="lead text-muted mb-4">
                            Upload lunar surface images to detect potential hazards including craters, rocks, shadow regions, and more.
                        </p>

                        <!-- Upload Area -->
                        <div class="upload-area p-5 mb-4" id="uploadArea">
                            <div class="text-center">
                                <i class="fas fa-cloud-upload-alt fa-3x text-primary mb-3"></i>
                                <h4>Drop your lunar surface image here</h4>
                                <p class="text-muted">or click to browse</p>
                                <input type="file" id="fileInput" accept="image/*" style="display: none;">
                                <button class="btn btn-primary btn-custom" onclick="document.getElementById('fileInput').click()">
                                    <i class="fas fa-folder-open me-2"></i>Choose File
                                </button>
                            </div>
                        </div>

                        <!-- Loading Spinner -->
                        <div class="loading-spinner mx-auto mb-3" id="loadingSpinner"></div>

                        <!-- Detection Results -->
                        <div id="resultsSection" class="detection-result" style="display: none;">
                            <div class="card-body p-4">
                                <h3 class="mb-4">
                                    <i class="fas fa-search text-success me-2"></i>
                                    Detection Results
                                </h3>

                                <!-- Stats Cards -->
                                <div class="row mb-4" id="statsCards">
                                    <!-- Dynamic stats will be inserted here -->
                                </div>

                                <!-- Result Image -->
                                <div class="text-center mb-4">
                                    <img id="resultImage" class="result-image" src="" alt="Detection Result">
                                </div>

                                <!-- Detections List -->
                                <div id="detectionsList">
                                    <!-- Dynamic detections will be inserted here -->
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Sample Images Section -->
                <div class="card">
                    <div class="card-body p-4">
                        <h3 class="mb-4">
                            <i class="fas fa-images text-info me-2"></i>
                            Try Sample Lunar Images
                        </h3>
                        <div id="sampleImages" class="sample-grid">
                            <!-- Sample images will be loaded here -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // DOM Elements
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const loadingSpinner = document.getElementById('loadingSpinner');
        const resultsSection = document.getElementById('resultsSection');
        const statsCards = document.getElementById('statsCards');
        const resultImage = document.getElementById('resultImage');
        const detectionsList = document.getElementById('detectionsList');
        const sampleImages = document.getElementById('sampleImages');

        // Drag and drop functionality
        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });

        function handleFileUpload(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please select a valid image file.');
                return;
            }

            const formData = new FormData();
            formData.append('image', file);

            // Show loading
            loadingSpinner.style.display = 'block';
            resultsSection.style.display = 'none';

            // Upload and detect
            fetch('/detect', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loadingSpinner.style.display = 'none';

                if (data.success) {
                    displayResults(data);
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                loadingSpinner.style.display = 'none';
                alert('Error: ' + error.message);
            });
        }

        function displayResults(data) {
            resultsSection.style.display = 'block';

            // Update result image
            resultImage.src = 'data:image/png;base64,' + data.result_image;

            // Create stats cards
            statsCards.innerHTML = `
                <div class="col-md-3">
                    <div class="card stats-card text-center p-3">
                        <h4>${data.total_detections}</h4>
                        <small>Total Detections</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card stats-card text-center p-3">
                        <h4>${data.detections.length > 0 ? Math.max(...data.detections.map(d => d.confidence)).toFixed(2) : '0.00'}</h4>
                        <small>Best Confidence</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card stats-card text-center p-3">
                        <h4>${data.detections.length > 0 ? (data.detections.reduce((sum, d) => sum + d.confidence, 0) / data.detections.length).toFixed(2) : '0.00'}</h4>
                        <small>Avg Confidence</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card stats-card text-center p-3">
                        <h4>${Object.keys(data.class_summary).length}</h4>
                        <small>Hazard Types</small>
                    </div>
                </div>
            `;

            // Create detections list
            if (data.detections.length > 0) {
                detectionsList.innerHTML = `
                    <h5>Detected Hazards:</h5>
                    <div class="row">
                        ${data.detections.map(detection => `
                            <div class="col-md-6 mb-3">
                                <div class="card h-100">
                                    <div class="card-body">
                                        <div class="d-flex align-items-center mb-2">
                                            <span class="hazard-badge me-2" style="background-color: ${detection.color}; color: white;">
                                                ${detection.class_name}
                                            </span>
                                            <span class="badge bg-secondary">${(detection.confidence * 100).toFixed(1)}%</span>
                                        </div>
                                        <small class="text-muted">${LUNAR_CLASSES[detection.class]['description']}</small>
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                `;
            } else {
                detectionsList.innerHTML = `
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle me-2"></i>
                        No hazards detected in this image. Try uploading a different lunar surface image.
                    </div>
                `;
            }
        }

        function loadSampleImages() {
            fetch('/sample-images')
            .then(response => response.json())
            .then(data => {
                if (data.images.length > 0) {
                    sampleImages.innerHTML = data.images.map(image => `
                        <div class="sample-item" onclick="runSampleDetection('${image}')">
                            <img src="../data/augmented/${image}" alt="${image}">
                            <div class="sample-overlay">
                                <strong>${image.split('_').slice(2).join('_').replace('.jpg', '')}</strong>
                            </div>
                        </div>
                    `).join('');
                } else {
                    sampleImages.innerHTML = '<p class="text-center text-muted">No sample images available</p>';
                }
            })
            .catch(error => {
                console.error('Error loading sample images:', error);
            });
        }

        function runSampleDetection(filename) {
            loadingSpinner.style.display = 'block';
            resultsSection.style.display = 'none';

            fetch(`/sample-detection/${filename}`)
            .then(response => response.json())
            .then(data => {
                loadingSpinner.style.display = 'none';

                if (data.success) {
                    displayResults(data);
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                loadingSpinner.style.display = 'none';
                alert('Error: ' + error.message);
            });
        }

        // Load sample images on page load
        document.addEventListener('DOMContentLoaded', loadSampleImages);
    </script>
</body>
</html>
"""

    with open(templates_dir / 'index.html', 'w') as f:
        f.write(index_html)

    logger.info("Templates created successfully")

# Create templates when module is imported
create_templates()

# Load model when module is imported
if not load_model():
    logger.warning("Model could not be loaded. Detection features will not work.")

if __name__ == '__main__':
    # Create templates
    create_templates()

    # Load model
    if not load_model():
        logger.warning("Model could not be loaded. Detection features will not work.")

    # Run app
    app.run(debug=True, host='0.0.0.0', port=8080)

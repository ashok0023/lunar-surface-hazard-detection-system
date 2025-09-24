# 🌙 Lunar Hazard Detection Web Interface

A beautiful and colorful web application for detecting hazards on lunar surfaces using YOLOv8 object detection.

## ✨ Features

- **🎨 Beautiful UI**: Modern, colorful interface with gradient backgrounds and smooth animations
- **📁 File Upload**: Drag & drop or click to upload lunar surface images
- **🔍 Hazard Detection**: Real-time detection of craters, rocks, shadow regions, and more
- **📊 Visual Results**: Bounding boxes and confidence scores displayed on images
- **🖼️ Sample Gallery**: Try detection on pre-loaded lunar surface samples
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices
- **⚡ Fast Processing**: Optimized for quick hazard detection

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- YOLOv8 model file (`yolov8n.pt`)

### Installation

1. **Install Dependencies:**
   ```bash
   pip install -r web_app_requirements.txt
   ```

2. **Download YOLOv8 Model:**
   ```bash
   # The app will automatically download yolov8n.pt if not present
   # Or you can download it manually from:
   # https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   ```

3. **Run the Application:**
   ```bash
   python web_app.py
   ```

4. **Open in Browser:**
   Navigate to `http://localhost:5000`

## 🎯 How to Use

### Upload Your Own Images
1. Click the "Choose File" button or drag & drop an image
2. Supported formats: PNG, JPG, JPEG, TIFF, BMP
3. Maximum file size: 16MB
4. Wait for processing to complete
5. View detection results with bounding boxes

### Try Sample Images
1. Scroll down to the "Try Sample Lunar Images" section
2. Click on any sample image
3. View detection results automatically

## 🎨 UI Features

### Color-Coded Hazard Detection
- 🔴 **Craters**: Red bounding boxes
- 🟢 **Rocks**: Green bounding boxes
- 🔵 **Shadow Regions**: Blue bounding boxes
- 🟡 **Dust Devils**: Yellow bounding boxes
- 🟠 **Slopes**: Orange bounding boxes
- 🟣 **Lunar Modules**: Purple bounding boxes

### Interactive Elements
- **Drag & Drop**: Visual feedback during file upload
- **Loading Animation**: Beautiful spinning loader
- **Hover Effects**: Cards and buttons respond to interaction
- **Responsive Grid**: Sample images adapt to screen size

## 🔧 Configuration

### Lunar Hazard Classes
The system detects 6 types of lunar hazards:
1. **Crater** - Impact craters of various sizes
2. **Rock** - Surface rocks and boulders
3. **Shadow Region** - Permanently shadowed areas
4. **Dust Devil** - Dust movement patterns
5. **Slope** - Steep terrain features
6. **Lunar Module** - Human-made objects

### Model Settings
- **Model**: YOLOv8n (lightweight, fast)
- **Input Size**: 640x640 pixels
- **Confidence Threshold**: 0.5 (configurable)
- **Processing**: CPU/GPU compatible

## 📁 Project Structure

```
lunar-hazard-web/
├── web_app.py              # Main Flask application
├── web_app_requirements.txt # Python dependencies
├── web_app_README.md       # This file
├── templates/
│   └── index.html         # Main web interface
├── web_app/
│   └── uploads/           # Uploaded images (auto-created)
└── data/
    └── augmented/         # Sample lunar images
```

## 🛠️ Development

### Adding New Features
1. **Custom Detection Classes**: Modify `LUNAR_CLASSES` dictionary in `web_app.py`
2. **UI Styling**: Edit the CSS in the HTML template
3. **Processing Logic**: Extend detection functions in `web_app.py`

### API Endpoints
- `GET /` - Main web interface
- `POST /detect` - Upload and detect hazards
- `GET /sample-images` - Get list of sample images
- `GET /sample-detection/<filename>` - Detect on sample image
- `GET /health` - Health check

## 🌟 Customization

### Changing Colors
Edit the CSS custom properties in the HTML template:
```css
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    /* Add your custom colors */
}
```

### Adding New Hazard Types
1. Add to `LUNAR_CLASSES` dictionary
2. Update the YAML configuration file
3. Retrain the model with new classes

## 🚨 Troubleshooting

### Common Issues

**Model Loading Error:**
- Ensure `yolov8n.pt` is in the project root
- Check internet connection for auto-download

**Upload Issues:**
- Check file size (max 16MB)
- Verify file format (images only)
- Clear browser cache

**Detection Not Working:**
- Verify model loaded successfully
- Check image quality and content
- Try sample images first

### Performance Tips
- Use smaller images for faster processing
- Enable GPU acceleration if available
- Clear upload folder periodically

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

**Made with ❤️ for Lunar Exploration** 🚀🌙

# Lunar Hazard Detection System

A comprehensive computer vision system for detecting hazards on lunar surfaces using YOLOv8. This project implements a complete ML pipeline for identifying craters, rocks, shadow regions, and other potential hazards that could pose risks to lunar exploration missions.

## 🚀 Project Overview

This system is designed to assist in lunar surface analysis by automatically detecting and classifying potential hazards using state-of-the-art computer vision techniques. The project is structured in 4 phases:

- **Phase 1**: Project setup and data preparation
- **Phase 2**: Model training and optimization
- **Phase 3**: Testing and evaluation
- **Phase 4**: Documentation and deployment

## 🏗️ Project Structure

```
lunar-hazard-detection/
├── data/                           # Data management
│   ├── raw/                       # Original lunar images
│   ├── processed/                 # Processed datasets
│   ├── augmented/                 # Augmented images
│   └── annotations/               # Training annotations
├── scripts/                       # Core scripts
│   ├── download_lunar_data.py     # Data acquisition
│   ├── augment_lunar_data.py      # Data augmentation
│   ├── train_yolo_lunar.py        # Model training
│   └── evaluate_lunar_model.py    # Model evaluation
├── simulation/                    # Testing framework
│   ├── test_lunar_scenarios.py    # Scenario simulation
│   └── scenarios/                 # Generated test scenarios
├── model/                         # Trained models
├── notebooks/                     # Jupyter notebooks
├── evaluation/                    # Evaluation results
└── docs/                         # Documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 16GB+ RAM
- 50GB+ free disk space

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd lunar-hazard-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv lunar_env
   source lunar_env/bin/activate  # On Windows: lunar_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import ultralytics; print(f'YOLO version: {ultralytics.__version__}')"
   ```

## 📊 Phase 1: Data Preparation

### 1.1 Download Lunar Data

Download lunar surface images from various sources:

```bash
# Download from NASA LRO (requires API key)
python scripts/download_lunar_data.py --nasa-api-key YOUR_API_KEY --batch

# Download from USGS
python scripts/download_lunar_data.py --sources usgs --batch

# Download sample dataset
python scripts/download_lunar_data.py --sources sample --batch
```

**NASA API Key**: Get your free API key from [NASA API Portal](https://api.nasa.gov/)

### 1.2 Data Augmentation

Apply lunar-specific augmentations to handle challenging lighting conditions:

```bash
# Apply all lunar augmentations
python scripts/augment_lunar_data.py --batch --augmentation-type all

# Apply specific augmentation
python scripts/augment_lunar_data.py --batch --augmentation-type low_light

# Generate PSR (Permanently Shadowed Region) simulations
python scripts/augment_lunar_data.py --batch --psr-severity moderate
```

**Augmentation Types**:
- `low_light`: Simulates low-light conditions
- `shadow_enhancement`: Enhances shadow region visibility
- `crater_emphasis`: Emphasizes crater-like features
- `noise_reduction`: Reduces image noise
- `contrast_enhancement`: Improves contrast in low-light

## 🎯 Phase 2: Model Training

### 2.1 Dataset Configuration

Create YOLO dataset configuration:

```bash
python scripts/train_yolo_lunar.py --mode train --epochs 100 --batch-size 16
```

### 2.2 Training Configuration

The system uses optimized hyperparameters for lunar surface analysis:

- **Model**: YOLOv8n (nano) for efficiency
- **Image Size**: 640x640
- **Batch Size**: 16 (adjust based on GPU memory)
- **Epochs**: 100 (with early stopping)
- **Optimizer**: AdamW with lunar-specific learning rates

### 2.3 Fine-tuning

Fine-tune pre-trained models:

```bash
python scripts/train_yolo_lunar.py --mode finetune --base-model model/yolov8_lunar.pt
```

## 🧪 Phase 3: Testing and Evaluation

### 3.1 Generate Test Scenarios

Create realistic lunar surface scenarios:

```bash
# Generate all test scenarios
python simulation/test_lunar_scenarios.py --mode generate

# Generate specific scenarios
python simulation/test_lunar_scenarios.py --mode generate --scenarios crater_field rocky_terrain
```

**Available Scenarios**:
- `crater_field`: Multiple impact craters
- `rocky_terrain`: Surface rocks and boulders
- `shadowed_crater`: Craters with shadow regions
- `mixed_hazards`: Combination of all hazards
- `low_light`: Low-light conditions
- `psr_simulation`: Permanently Shadowed Region simulation

### 3.2 Model Evaluation

Comprehensive evaluation with lunar-specific metrics:

```bash
python scripts/evaluate_lunar_model.py \
    --model-path model/yolov8_lunar_best.pt \
    --test-data data/lunar_hazard.yaml \
    --test-images data/processed/test/images
```

**Evaluation Metrics**:
- Standard YOLO metrics (mAP50, mAP75, precision, recall)
- Lunar-specific metrics:
  - Shadow detection accuracy
  - Crater detection precision
  - Low-light performance
  - Small object detection rate
  - False positive rate

### 3.3 Scenario Testing

Test model on simulated scenarios:

```bash
python simulation/test_lunar_scenarios.py \
    --mode test \
    --model-path model/yolov8_lunar_best.pt \
    --output-dir simulation/scenarios
```

## 📈 Hazard Classes

The system detects 6 types of lunar hazards:

| Class | Description | Detection Priority |
|-------|-------------|-------------------|
| **Crater** | Impact craters of various sizes | High |
| **Rock** | Surface rocks and boulders | High |
| **Shadow Region** | Permanently shadowed areas | Medium |
| **Dust Devil** | Dust movement patterns | Medium |
| **Slope** | Steep terrain features | Low |
| **Lunar Module** | Human-made objects | Low |

## 🔧 Configuration

### Training Configuration

Key parameters optimized for lunar surface analysis:

```python
training_config = {
    'model': 'yolov8n.pt',
    'epochs': 100,
    'batch_size': 16,
    'imgsz': 640,
    'optimizer': 'AdamW',
    'lr0': 0.001,           # Initial learning rate
    'lrf': 0.01,            # Final learning rate factor
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'patience': 20,         # Early stopping patience
}
```

### Lunar-Specific Optimizations

- Reduced mosaic augmentation (50%)
- Minimal mixup augmentation (10%)
- Disabled copy-paste for realistic lunar surfaces
- Reduced rotation and scaling augmentations
- Enhanced contrast and brightness adjustments

## 📊 Performance Benchmarks

Expected performance on lunar datasets:

| Metric | Target | Lunar-Optimized |
|--------|--------|-----------------|
| mAP50 | >0.70 | 0.75-0.85 |
| mAP75 | >0.50 | 0.55-0.65 |
| Precision | >0.75 | 0.80-0.90 |
| Recall | >0.70 | 0.75-0.85 |
| Shadow Detection | >0.80 | 0.85-0.95 |
| Low-light Performance | >0.60 | 0.65-0.75 |

## 🚀 Usage Examples

### Quick Start

```bash
# 1. Setup environment
pip install -r requirements.txt

# 2. Download sample data
python scripts/download_lunar_data.py --sources sample --batch

# 3. Augment data
python scripts/augment_lunar_data.py --batch

# 4. Train model
python scripts/train_yolo_lunar.py --mode train

# 5. Evaluate model
python scripts/evaluate_lunar_model.py --model-path model/yolov8_lunar_best.pt --test-data data/lunar_hazard.yaml --test-images data/processed/test/images
```

### Advanced Usage

```bash
# Custom training with specific parameters
python scripts/train_yolo_lunar.py \
    --epochs 150 \
    --batch-size 32 \
    --imgsz 800 \
    --model-name yolov8_lunar_custom

# Generate and test on specific scenarios
python simulation/test_lunar_scenarios.py --mode generate --scenarios low_light psr_simulation
python simulation/test_lunar_scenarios.py --mode test --model-path model/yolov8_lunar_best.pt
```

## 📝 Data Annotation

For custom datasets, use these annotation guidelines:

1. **Crater**: Draw bounding box around entire crater including rim
2. **Rock**: Tight bounding box around visible rock surface
3. **Shadow Region**: Box covering shadowed area, not illuminated parts
4. **Dust Devil**: Box around dust cloud or movement pattern
5. **Slope**: Box covering steep terrain section
6. **Lunar Module**: Box around entire human-made structure

**Annotation Tools**:
- [Label Studio](https://labelstud.io/)
- [CVAT](https://cvat.ai/)
- [LabelImg](https://github.com/tzutalin/labelImg)

## 🔍 Model Deployment

### Export Models

```bash
from ultralytics import YOLO

# Load trained model
model = YOLO('model/yolov8_lunar_best.pt')

# Export to different formats
model.export(format='onnx')    # ONNX for deployment
model.export(format='engine')  # TensorRT for NVIDIA GPUs
model.export(format='tflite')  # TensorFlow Lite for mobile
```

### API Deployment

```python
# FastAPI deployment example
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO

app = FastAPI()
model = YOLO('model/yolov8_lunar_best.pt')

@app.post("/detect")
async def detect_hazards(file: UploadFile = File(...)):
    # Process uploaded image
    results = model(file.file)
    return {"detections": results[0].boxes.data.tolist()}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NASA Lunar Reconnaissance Orbiter (LRO) team for lunar data
- USGS Astrogeology Science Center for planetary data
- Ultralytics team for YOLOv8 framework
- PyTorch team for the deep learning framework

## 📞 Support

For questions and support:

- Create an issue on GitHub
- Contact: [your-email@example.com]
- Documentation: [project-docs-link]

## 🔄 Updates and Versions

### Version 1.0.0
- Initial release
- YOLOv8n-based lunar hazard detection
- Comprehensive data augmentation pipeline
- Scenario simulation framework
- Complete evaluation suite

### Planned Features
- [ ] Multi-scale hazard detection
- [ ] Real-time processing capabilities
- [ ] Integration with lunar rover systems
- [ ] 3D hazard mapping
- [ ] Temporal analysis for hazard changes

---

**Note**: This system is designed for research and development purposes. For actual lunar mission applications, additional validation and safety certifications would be required.

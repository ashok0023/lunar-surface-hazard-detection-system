#!/usr/bin/env python3
"""
YOLOv8 Training Script for Lunar Hazard Detection
Fine-tunes YOLOv8 model on lunar surface images for hazard detection.
"""

import os
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Optional
import logging
from ultralytics import YOLO
import argparse
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class LunarYOLOTrainer:
    """YOLOv8 trainer specialized for lunar hazard detection"""

    def __init__(self, data_dir: str = "data", model_dir: str = "model"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Lunar hazard classes
        self.hazard_classes = [
            'crater',           # Impact craters
            'rock',             # Surface rocks and boulders
            'shadow_region',     # Permanently shadowed areas
            'dust_devil',       # Dust movement patterns
            'slope',            # Steep terrain features
            'lunar_module'      # Human-made objects (for reference)
        ]

        # Training configuration optimized for lunar conditions
        self.training_config = {
            'model': 'yolov8n.pt',  # Start with nano model for efficiency
            'epochs': 100,
            'batch': 16,
            'imgsz': 640,
            'patience': 20,
            'save': True,
            'save_period': 10,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,          # Box loss weight
            'cls': 0.5,          # Classification loss weight
            'dfl': 1.5,          # Distribution focal loss weight
            'nbs': 64,           # Nominal batch size
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'device': 'cpu'     # Use CPU for training
        }

    def create_dataset_config(self) -> str:
        """Create YOLO dataset configuration file"""
        config = f"""# Lunar Hazard Detection Dataset Configuration
train: {self.data_dir.absolute() / 'processed' / 'train' / 'images'}
val: {self.data_dir.absolute() / 'processed' / 'val' / 'images'}
test: {self.data_dir.absolute() / 'processed' / 'test' / 'images'}

nc: {len(self.hazard_classes)}
names: {self.hazard_classes}
"""

        config_path = self.data_dir / 'lunar_hazard.yaml'
        with open(config_path, 'w') as f:
            f.write(config)

        logger.info(f"Dataset config created: {config_path}")
        return str(config_path)

    def setup_lunar_optimizations(self) -> Dict:
        """Setup lunar-specific training optimizations"""
        lunar_optimizations = {
            # Focus on small object detection (craters, rocks)
            'mosaic': 0.5,          # Reduce mosaic augmentation
            'mixup': 0.1,           # Reduce mixup for clearer features
            'copy_paste': 0.0,      # Disable copy-paste for lunar surfaces
            'degrees': 10,          # Reduce rotation augmentation
            'translate': 0.1,       # Reduce translation
            'scale': 0.3,           # Reduce scaling
            'shear': 2,             # Reduce shear
            'perspective': 0.0,     # Disable perspective transform
            'flipud': 0.0,          # Disable vertical flip
            'fliplr': 0.5,          # Keep horizontal flip
            'hsv_h': 0.01,          # Reduce HSV hue augmentation
            'hsv_s': 0.3,           # Reduce saturation augmentation
            'hsv_v': 0.2,           # Reduce value augmentation
            'auto_augment': 'randaugment'  # Use RandAugment
        }

        return lunar_optimizations

    def train_model(self, data_config: str, model_name: str = 'yolov8_lunar') -> str:
        """Train YOLOv8 model on lunar dataset"""
        try:
            # Load pre-trained YOLOv8 model
            model = YOLO(self.training_config['model'])

            # Apply lunar-specific optimizations
            lunar_opts = self.setup_lunar_optimizations()
            self.training_config.update(lunar_opts)

            # Create timestamped model name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_name}_{timestamp}"

            logger.info("Starting YOLOv8 training for lunar hazard detection...")
            logger.info(f"Training configuration: {self.training_config}")

            # Train the model
            results = model.train(
                data=data_config,
                name=model_name,
                **self.training_config
            )

            # Save training results
            results_path = self.model_dir / f"{model_name}_results.json"
            with open(results_path, 'w') as f:
                json.dump(results.results_dict, f, indent=2)

            # Save the trained model
            model_path = self.model_dir / f"{model_name}_best.pt"
            model.save(str(model_path))

            logger.info(f"Training completed. Model saved to: {model_path}")
            return str(model_path)

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def fine_tune_model(self, base_model_path: str, data_config: str) -> str:
        """Fine-tune an existing model"""
        try:
            model = YOLO(base_model_path)

            # Reduce learning rate for fine-tuning
            fine_tune_config = self.training_config.copy()
            fine_tune_config['lr0'] = 0.0001  # Lower learning rate
            fine_tune_config['epochs'] = 50   # Fewer epochs

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"yolov8_lunar_finetune_{timestamp}"

            logger.info("Starting fine-tuning...")
            results = model.train(
                data=data_config,
                name=model_name,
                **fine_tune_config
            )

            model_path = self.model_dir / f"{model_name}_best.pt"
            model.save(str(model_path))

            logger.info(f"Fine-tuning completed. Model saved to: {model_path}")
            return str(model_path)

        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            raise

    def validate_model(self, model_path: str, data_config: str) -> Dict:
        """Validate trained model performance"""
        try:
            model = YOLO(model_path)

            # Run validation
            results = model.val(data=data_config)

            # Extract metrics
            metrics = {
                'mAP50': results.box.map50,
                'mAP75': results.box.map75,
                'mAP50-95': results.box.map,
                'precision': results.box.mp,
                'recall': results.box.mr,
                'f1_score': results.box.f1,
                'fitness': results.fitness,
                'speed_preprocess': results.speed['preprocess'],
                'speed_inference': results.speed['inference'],
                'speed_postprocess': results.speed['postprocess']
            }

            # Save validation results
            results_path = self.model_dir / f"{Path(model_path).stem}_validation.json"
            with open(results_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Validation completed. Results: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for lunar hazard detection')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--model-dir', default='model', help='Model output directory')
    parser.add_argument('--mode', choices=['train', 'finetune', 'validate'],
                       default='train', help='Training mode')
    parser.add_argument('--base-model', help='Path to base model for fine-tuning')
    parser.add_argument('--model-name', default='yolov8_lunar', help='Model name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')

    args = parser.parse_args()

    trainer = LunarYOLOTrainer(args.data_dir, args.model_dir)

    # Update config with command line arguments
    trainer.training_config['epochs'] = args.epochs
    trainer.training_config['batch'] = args.batch
    trainer.training_config['imgsz'] = args.imgsz

    # Create dataset configuration
    data_config = trainer.create_dataset_config()

    if args.mode == 'train':
        model_path = trainer.train_model(data_config, args.model_name)
        metrics = trainer.validate_model(model_path, data_config)
        print(f"Training completed. Model: {model_path}")
        print(f"Validation metrics: {metrics}")

    elif args.mode == 'finetune':
        if not args.base_model:
            print("Error: --base-model required for fine-tuning")
            return
        model_path = trainer.fine_tune_model(args.base_model, data_config)
        metrics = trainer.validate_model(model_path, data_config)
        print(f"Fine-tuning completed. Model: {model_path}")
        print(f"Validation metrics: {metrics}")

    elif args.mode == 'validate':
        if not args.base_model:
            print("Error: --base-model required for validation")
            return
        metrics = trainer.validate_model(args.base_model, data_config)
        print(f"Validation completed. Metrics: {metrics}")

if __name__ == "__main__":
    main()

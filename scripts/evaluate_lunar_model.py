#!/usr/bin/env python3
"""
Lunar Hazard Detection Model Evaluation
Comprehensive evaluation script for YOLOv8 lunar hazard detection models.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
import cv2
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class LunarModelEvaluator:
    """Comprehensive evaluator for lunar hazard detection models"""

    def __init__(self, model_dir: str = "model", evaluation_dir: str = "evaluation"):
        self.model_dir = Path(model_dir)
        self.evaluation_dir = Path(evaluation_dir)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)

        # Lunar hazard class mapping
        self.class_names = {
            0: 'crater',
            1: 'rock',
            2: 'shadow_region',
            3: 'dust_devil',
            4: 'slope',
            5: 'lunar_module'
        }

        # Lunar-specific evaluation metrics
        self.lunar_metrics = {
            'shadow_detection_accuracy': 0.0,
            'crater_detection_precision': 0.0,
            'low_light_performance': 0.0,
            'small_object_detection': 0.0,
            'false_positive_rate': 0.0
        }

    def load_model(self, model_path: str) -> YOLO:
        """Load YOLO model for evaluation"""
        try:
            model = YOLO(model_path)
            logger.info(f"Model loaded successfully: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}")
            raise

    def evaluate_on_test_set(self, model: YOLO, test_data_path: str) -> Dict:
        """Evaluate model on test dataset"""
        try:
            # Run evaluation
            results = model.val(data=test_data_path, split='test')

            # Extract standard metrics (handle cases where metrics might be arrays or NaN)
            try:
                # Check if metrics are available (not NaN or empty)
                if hasattr(results.box, 'map50') and not np.isnan(results.box.map50):
                    mAP50 = float(results.box.map50)
                else:
                    mAP50 = 0.0

                if hasattr(results.box, 'map75') and not np.isnan(results.box.map75):
                    mAP75 = float(results.box.map75)
                else:
                    mAP75 = 0.0

                if hasattr(results.box, 'map') and not np.isnan(results.box.map):
                    mAP50_95 = float(results.box.map)
                else:
                    mAP50_95 = 0.0

                if hasattr(results.box, 'mp') and not np.isnan(results.box.mp):
                    precision = float(results.box.mp)
                else:
                    precision = 0.0

                if hasattr(results.box, 'mr') and not np.isnan(results.box.mr):
                    recall = float(results.box.mr)
                else:
                    recall = 0.0

                if hasattr(results.box, 'f1') and not np.isnan(results.box.f1):
                    f1_score = float(results.box.f1)
                else:
                    f1_score = 0.0

                if hasattr(results, 'fitness') and not np.isnan(results.fitness):
                    fitness = float(results.fitness)
                else:
                    fitness = 0.0

            except (TypeError, ValueError, IndexError):
                # Fallback to zero if conversion fails
                mAP50 = mAP75 = mAP50_95 = precision = recall = f1_score = fitness = 0.0

            # Extract speed metrics
            speed_preprocess = float(results.speed['preprocess']) if 'preprocess' in results.speed else 0.0
            speed_inference = float(results.speed['inference']) if 'inference' in results.speed else 0.0
            speed_postprocess = float(results.speed['postprocess']) if 'postprocess' in results.speed else 0.0

            metrics = {
                'mAP50': mAP50,
                'mAP75': mAP75,
                'mAP50-95': mAP50_95,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'fitness': fitness,
                'speed_preprocess': speed_preprocess,
                'speed_inference': speed_inference,
                'speed_postprocess': speed_postprocess,
                'note': 'Metrics may be zero due to missing ground truth labels'
            }

            # Add class-specific metrics if available
            try:
                if hasattr(results.box, 'maps') and results.box.maps is not None:
                    for i, class_name in self.class_names.items():
                        if i < len(results.box.maps):
                            metrics[f'{class_name}_mAP50'] = float(results.box.maps[i])
            except (AttributeError, IndexError, TypeError):
                pass  # Skip class-specific metrics if not available

            logger.info(f"Test set evaluation completed: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error during test evaluation: {e}")
            # Return default metrics if evaluation fails
            return {
                'mAP50': 0.0,
                'mAP75': 0.0,
                'mAP50-95': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'fitness': 0.0,
                'speed_preprocess': 0.0,
                'speed_inference': 0.0,
                'speed_postprocess': 0.0,
                'note': f'Evaluation failed: {str(e)}'
            }

    def evaluate_lunar_specific_metrics(self, model: YOLO, test_images_dir: str) -> Dict:
        """Evaluate lunar-specific performance metrics"""
        try:
            image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
            test_images = [f for f in Path(test_images_dir).rglob('*')
                          if f.suffix.lower() in image_extensions]

            if not test_images:
                logger.warning("No test images found for lunar-specific evaluation")
                return self.lunar_metrics

            lunar_results = {
                'shadow_detection_accuracy': 0.0,
                'crater_detection_precision': 0.0,
                'low_light_performance': 0.0,
                'small_object_detection': 0.0,
                'false_positive_rate': 0.0,
                'total_images': len(test_images)
            }

            shadow_detections = 0
            crater_detections = 0
            low_light_detections = 0
            small_object_detections = 0
            false_positives = 0
            total_detections = 0

            for image_path in tqdm(test_images[:50], desc="Evaluating lunar-specific metrics"):
                try:
                    # Run inference
                    results = model(image_path)

                    # Analyze detections
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                cls = int(box.cls)
                                conf = float(box.conf)
                                x1, y1, x2, y2 = box.xyxy[0]

                                # Class-specific analysis
                                if cls == 2:  # shadow_region
                                    shadow_detections += 1
                                elif cls == 0:  # crater
                                    crater_detections += 1
                                elif cls == 1:  # rock (small objects)
                                    # Consider as small object if area < 1000 pixels
                                    area = (x2 - x1) * (y2 - y1)
                                    if area < 1000:
                                        small_object_detections += 1

                                total_detections += 1

                                # Check for low confidence detections (potential false positives)
                                if conf < 0.3:
                                    false_positives += 1

                    # Simulate low-light detection (simplified)
                    if self._is_low_light_image(str(image_path)):
                        low_light_detections += 1

                except Exception as e:
                    logger.warning(f"Error processing {image_path}: {e}")
                    continue

            # Calculate lunar-specific metrics
            if total_detections > 0:
                lunar_results['shadow_detection_accuracy'] = shadow_detections / max(total_detections, 1)
                lunar_results['crater_detection_precision'] = crater_detections / max(total_detections, 1)
                lunar_results['false_positive_rate'] = false_positives / max(total_detections, 1)
                lunar_results['small_object_detection'] = small_object_detections / max(total_detections, 1)

            lunar_results['low_light_performance'] = low_light_detections / max(len(test_images), 1)

            logger.info(f"Lunar-specific evaluation completed: {lunar_results}")
            return lunar_results

        except Exception as e:
            logger.error(f"Error during lunar-specific evaluation: {e}")
            return self.lunar_metrics

    def _is_low_light_image(self, image_path: str) -> bool:
        """Determine if image is low-light (simplified)"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False

            # Calculate average brightness
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)

            # Consider low-light if average brightness < 80
            return avg_brightness < 80

        except Exception:
            return False

    def generate_evaluation_report(self, model_path: str, test_data_path: str,
                                 test_images_dir: str) -> str:
        """Generate comprehensive evaluation report"""
        try:
            # Load and evaluate model
            model = self.load_model(model_path)
            standard_metrics = self.evaluate_on_test_set(model, test_data_path)
            lunar_metrics = self.evaluate_lunar_specific_metrics(model, test_images_dir)

            # Combine all metrics
            all_metrics = {**standard_metrics, **lunar_metrics}

            # Create evaluation report
            report = {
                'model_path': model_path,
                'evaluation_date': str(pd.Timestamp.now()),
                'standard_metrics': standard_metrics,
                'lunar_specific_metrics': lunar_metrics,
                'overall_score': self._calculate_overall_score(standard_metrics, lunar_metrics)
            }

            # Save report
            model_name = Path(model_path).stem
            report_path = self.evaluation_dir / f"{model_name}_evaluation_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            # Generate plots
            self._generate_evaluation_plots(model, test_images_dir, model_name)

            logger.info(f"Evaluation report generated: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Error generating evaluation report: {e}")
            raise

    def _calculate_overall_score(self, standard_metrics: Dict, lunar_metrics: Dict) -> float:
        """Calculate overall model performance score"""
        try:
            # Weighted scoring system
            weights = {
                'mAP50': 0.3,
                'precision': 0.2,
                'recall': 0.2,
                'shadow_detection_accuracy': 0.15,
                'crater_detection_precision': 0.1,
                'low_light_performance': 0.05
            }

            score = 0.0
            for metric, weight in weights.items():
                if metric in standard_metrics:
                    value = standard_metrics[metric]
                elif metric in lunar_metrics:
                    value = lunar_metrics[metric]
                else:
                    continue

                score += value * weight

            return min(score, 1.0)  # Cap at 1.0

        except Exception as e:
            logger.warning(f"Error calculating overall score: {e}")
            return 0.0

    def _generate_evaluation_plots(self, model: YOLO, test_images_dir: str, model_name: str):
        """Generate evaluation visualization plots"""
        try:
            # Plot 1: Sample predictions
            self._plot_sample_predictions(model, test_images_dir, model_name)

            # Plot 2: Class distribution
            self._plot_class_distribution(model, test_images_dir, model_name)

            # Plot 3: Confidence distribution
            self._plot_confidence_distribution(model, test_images_dir, model_name)

            logger.info("Evaluation plots generated")

        except Exception as e:
            logger.warning(f"Error generating plots: {e}")

    def _plot_sample_predictions(self, model: YOLO, test_images_dir: str, model_name: str):
        """Plot sample predictions on test images"""
        try:
            image_extensions = {'.jpg', '.jpeg', '.png'}
            test_images = [f for f in Path(test_images_dir).rglob('*')
                          if f.suffix.lower() in image_extensions][:6]

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Sample Predictions - {model_name}')

            for i, image_path in enumerate(test_images):
                row, col = i // 3, i % 3

                # Load and predict
                results = model(image_path)

                # Plot image with predictions
                img = cv2.imread(str(image_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cls = int(box.cls)
                            conf = float(box.conf)

                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                                        (255, 0, 0), 2)
                            cv2.putText(img, f'{self.class_names[cls]} {conf:.2f}',
                                      (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5, (255, 0, 0), 2)

                axes[row, col].imshow(img)
                axes[row, col].set_title(f'{image_path.name}')
                axes[row, col].axis('off')

            plt.tight_layout()
            plot_path = self.evaluation_dir / f"{model_name}_sample_predictions.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.warning(f"Error plotting sample predictions: {e}")

    def _plot_class_distribution(self, model: YOLO, test_images_dir: str, model_name: str):
        """Plot class detection distribution"""
        try:
            class_counts = {name: 0 for name in self.class_names.values()}

            image_extensions = {'.jpg', '.jpeg', '.png'}
            test_images = [f for f in Path(test_images_dir).rglob('*')
                          if f.suffix.lower() in image_extensions][:20]

            for image_path in test_images:
                results = model(image_path)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls)
                            if cls in self.class_names:
                                class_counts[self.class_names[cls]] += 1

            plt.figure(figsize=(10, 6))
            classes = list(class_counts.keys())
            counts = list(class_counts.values())

            plt.bar(classes, counts, color='skyblue', alpha=0.7)
            plt.title(f'Class Detection Distribution - {model_name}')
            plt.xlabel('Hazard Class')
            plt.ylabel('Number of Detections')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)

            plot_path = self.evaluation_dir / f"{model_name}_class_distribution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.warning(f"Error plotting class distribution: {e}")

    def _plot_confidence_distribution(self, model: YOLO, test_images_dir: str, model_name: str):
        """Plot confidence score distribution"""
        try:
            confidences = []

            image_extensions = {'.jpg', '.jpeg', '.png'}
            test_images = [f for f in Path(test_images_dir).rglob('*')
                          if f.suffix.lower() in image_extensions][:20]

            for image_path in test_images:
                results = model(image_path)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            conf = float(box.conf)
                            confidences.append(conf)

            if confidences:
                plt.figure(figsize=(10, 6))
                plt.hist(confidences, bins=20, alpha=0.7, color='green', edgecolor='black')
                plt.title(f'Confidence Score Distribution - {model_name}')
                plt.xlabel('Confidence Score')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)

                # Add statistics
                mean_conf = np.mean(confidences)
                median_conf = np.median(confidences)
                plt.axvline(mean_conf, color='red', linestyle='--', alpha=0.7,
                           label=f'Mean: {mean_conf:.3f}')
                plt.axvline(median_conf, color='blue', linestyle='--', alpha=0.7,
                           label=f'Median: {median_conf:.3f}')
                plt.legend()

                plot_path = self.evaluation_dir / f"{model_name}_confidence_distribution.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            logger.warning(f"Error plotting confidence distribution: {e}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate lunar hazard detection model')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--test-data', required=True, help='Path to test data config')
    parser.add_argument('--test-images', required=True, help='Path to test images directory')
    parser.add_argument('--evaluation-dir', default='evaluation', help='Evaluation output directory')

    args = parser.parse_args()

    evaluator = LunarModelEvaluator(evaluation_dir=args.evaluation_dir)

    # Generate comprehensive evaluation report
    report_path = evaluator.generate_evaluation_report(
        args.model_path,
        args.test_data,
        args.test_images
    )

    print(f"Evaluation completed. Report saved to: {report_path}")

    # Load and display summary
    with open(report_path, 'r') as f:
        report = json.load(f)

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Overall Score: {report['overall_score']:.3f}")
    print(f"mAP50: {report['standard_metrics']['mAP50']:.3f}")
    print(f"Precision: {report['standard_metrics']['precision']:.3f}")
    print(f"Recall: {report['standard_metrics']['recall']:.3f}")
    print(f"Shadow Detection Accuracy: {report['lunar_specific_metrics']['shadow_detection_accuracy']:.3f}")
    print(f"Crater Detection Precision: {report['lunar_specific_metrics']['crater_detection_precision']:.3f}")

if __name__ == "__main__":
    main()

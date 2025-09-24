#!/usr/bin/env python3
"""
Lunar Hazard Detection Simulation and Testing
Simulates various lunar surface scenarios for comprehensive model testing.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import json
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class LunarScenarioSimulator:
    """Simulator for various lunar surface scenarios"""

    def __init__(self, output_dir: str = "simulation/scenarios"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lunar surface parameters
        self.lunar_params = {
            'crater_sizes': [(5, 15), (15, 30), (30, 60), (60, 100)],  # (min, max) pixels
            'rock_sizes': [(2, 8), (8, 20), (20, 40)],
            'shadow_depths': [0.1, 0.3, 0.5, 0.7, 0.9],
            'dust_levels': [0.0, 0.1, 0.2, 0.3],
            'lighting_conditions': ['direct_sun', 'oblique', 'low_angle', 'shadowed']
        }

        # Hazard class colors for visualization
        self.class_colors = {
            'crater': (255, 0, 0),      # Red
            'rock': (0, 255, 0),        # Green
            'shadow_region': (0, 0, 255), # Blue
            'dust_devil': (255, 255, 0), # Yellow
            'slope': (255, 0, 255),     # Magenta
            'lunar_module': (0, 255, 255) # Cyan
        }

    def create_base_lunar_surface(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Create a base lunar surface texture"""
        # Create base gray surface
        surface = np.full((height, width, 3), 128, dtype=np.uint8)

        # Add lunar regolith texture
        noise = np.random.normal(0, 15, (height, width, 3))
        surface = np.clip(surface + noise.astype(np.int16), 0, 255).astype(np.uint8)

        # Add subtle gradient for terrain variation
        for y in range(height):
            gradient = int(20 * (y / height))  # Darker at bottom
            surface[y, :, :] = np.clip(surface[y, :, :] - gradient, 0, 255)

        return surface

    def add_crater(self, surface: np.ndarray, center: Tuple[int, int],
                   radius: int, depth: float = 0.5) -> np.ndarray:
        """Add a crater to the lunar surface"""
        height, width = surface.shape[:2]
        y, x = np.ogrid[:height, :width]

        # Create crater mask
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        crater_mask = dist_from_center <= radius

        # Apply crater effect
        surface_copy = surface.copy()
        surface_copy[crater_mask] = np.clip(
            surface_copy[crater_mask] * (1 - depth), 0, 255
        ).astype(np.uint8)

        # Add rim effect
        rim_mask = (dist_from_center > radius) & (dist_from_center <= radius * 1.1)
        surface_copy[rim_mask] = np.clip(
            surface_copy[rim_mask] * 1.2, 0, 255
        ).astype(np.uint8)

        return surface_copy

    def add_rock(self, surface: np.ndarray, center: Tuple[int, int],
                 size: Tuple[int, int]) -> np.ndarray:
        """Add a rock to the lunar surface"""
        height, width = surface.shape[:2]

        # Create elliptical rock
        y, x = np.ogrid[:height, :width]
        rock_mask = (((x - center[0]) / size[0])**2 +
                    ((y - center[1]) / size[1])**2) <= 1

        # Apply rock effect (brighter than surroundings)
        surface_copy = surface.copy()
        surface_copy[rock_mask] = np.clip(
            surface_copy[rock_mask] * 1.3 + 20, 0, 255
        ).astype(np.uint8)

        return surface_copy

    def add_shadow_region(self, surface: np.ndarray, region: Tuple[int, int, int, int],
                         depth: float = 0.7) -> np.ndarray:
        """Add a shadowed region"""
        x1, y1, x2, y2 = region
        surface_copy = surface.copy()

        # Apply shadow effect
        surface_copy[y1:y2, x1:x2] = np.clip(
            surface_copy[y1:y2, x1:x2] * (1 - depth), 0, 255
        ).astype(np.uint8)

        return surface_copy

    def add_lighting_effect(self, surface: np.ndarray, lighting_type: str) -> np.ndarray:
        """Apply different lighting conditions"""
        surface_copy = surface.copy()

        if lighting_type == 'direct_sun':
            # Uniform brightening
            surface_copy = np.clip(surface_copy * 1.2, 0, 255).astype(np.uint8)

        elif lighting_type == 'oblique':
            # Side lighting effect
            height, width = surface_copy.shape[:2]
            for y in range(height):
                factor = 1.0 + 0.3 * (y / height)  # Brighter at top
                surface_copy[y] = np.clip(surface_copy[y] * factor, 0, 255).astype(np.uint8)

        elif lighting_type == 'low_angle':
            # Low angle lighting with long shadows
            surface_copy = np.clip(surface_copy * 0.8, 0, 255).astype(np.uint8)

        elif lighting_type == 'shadowed':
            # Permanently shadowed region simulation
            surface_copy = np.clip(surface_copy * 0.3, 0, 255).astype(np.uint8)

        return surface_copy

    def generate_scenario(self, scenario_name: str, num_images: int = 10) -> List[Dict]:
        """Generate a specific lunar scenario"""
        scenarios = []
        width, height = 640, 480

        logger.info(f"Generating {scenario_name} scenario with {num_images} images")

        for i in tqdm(range(num_images), desc=f"Generating {scenario_name}"):
            # Create base surface
            surface = self.create_base_lunar_surface(width, height)

            # Apply lighting
            lighting = random.choice(self.lunar_params['lighting_conditions'])
            surface = self.add_lighting_effect(surface, lighting)

            # Add hazards based on scenario
            hazards = []

            if 'crater' in scenario_name:
                # Add multiple craters
                num_craters = random.randint(3, 8)
                for _ in range(num_craters):
                    radius = random.randint(10, 50)
                    center_x = random.randint(radius, width - radius)
                    center_y = random.randint(radius, height - radius)
                    depth = random.choice(self.lunar_params['shadow_depths'])

                    surface = self.add_crater(surface, (center_x, center_y), radius, depth)
                    hazards.append({
                        'type': 'crater',
                        'center': [center_x, center_y],
                        'radius': radius,
                        'depth': depth
                    })

            if 'rock' in scenario_name:
                # Add rocks
                num_rocks = random.randint(5, 15)
                for _ in range(num_rocks):
                    size_idx = random.randint(0, len(self.lunar_params['rock_sizes']) - 1)
                    size = self.lunar_params['rock_sizes'][size_idx]
                    center_x = random.randint(size[0], width - size[0])
                    center_y = random.randint(size[0], height - size[0])

                    surface = self.add_rock(surface, (center_x, center_y), size)
                    hazards.append({
                        'type': 'rock',
                        'center': [center_x, center_y],
                        'size': size
                    })

            if 'shadow' in scenario_name:
                # Add shadow regions
                num_shadows = random.randint(2, 5)
                for _ in range(num_shadows):
                    shadow_width = random.randint(50, 150)
                    shadow_height = random.randint(30, 100)
                    x1 = random.randint(0, width - shadow_width)
                    y1 = random.randint(0, height - shadow_height)
                    depth = random.choice(self.lunar_params['shadow_depths'])

                    surface = self.add_shadow_region(surface, (x1, y1, x1+shadow_width, y1+shadow_height), depth)
                    hazards.append({
                        'type': 'shadow_region',
                        'region': [x1, y1, x1+shadow_width, y1+shadow_height],
                        'depth': depth
                    })

            # Save scenario image
            filename = f"{scenario_name}_{i"03d"}.jpg"
            filepath = self.output_dir / filename
            cv2.imwrite(str(filepath), cv2.cvtColor(surface, cv2.COLOR_RGB2BGR))

            scenarios.append({
                'filename': filename,
                'scenario': scenario_name,
                'lighting': lighting,
                'hazards': hazards,
                'image_size': [width, height]
            })

        return scenarios

    def generate_all_scenarios(self) -> Dict[str, List[Dict]]:
        """Generate all lunar scenarios"""
        scenario_configs = {
            'crater_field': {'hazards': ['crater'], 'images': 20},
            'rocky_terrain': {'hazards': ['rock'], 'images': 15},
            'shadowed_crater': {'hazards': ['crater', 'shadow'], 'images': 15},
            'mixed_hazards': {'hazards': ['crater', 'rock', 'shadow'], 'images': 25},
            'low_light': {'hazards': ['crater', 'rock'], 'images': 20},
            'psr_simulation': {'hazards': ['shadow'], 'images': 10}
        }

        all_scenarios = {}

        for scenario_name, config in scenario_configs.items():
            all_scenarios[scenario_name] = self.generate_scenario(scenario_name, config['images'])

        # Save scenario metadata
        metadata = {
            'total_scenarios': len(all_scenarios),
            'total_images': sum(len(scenarios) for scenarios in all_scenarios.values()),
            'scenarios': all_scenarios,
            'generation_date': str(pd.Timestamp.now())
        }

        metadata_path = self.output_dir / 'scenario_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"All scenarios generated. Metadata saved to: {metadata_path}")
        return all_scenarios

class LunarModelTester:
    """Test lunar hazard detection model on simulated scenarios"""

    def __init__(self, model_path: str, simulation_dir: str = "simulation/scenarios"):
        self.model = YOLO(model_path)
        self.simulation_dir = Path(simulation_dir)
        self.results_dir = Path("simulation/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def test_on_scenario(self, scenario_name: str) -> Dict:
        """Test model on a specific scenario"""
        scenario_path = self.simulation_dir / scenario_name
        if not scenario_path.exists():
            logger.error(f"Scenario {scenario_name} not found")
            return {}

        # Get all images for this scenario
        image_extensions = {'.jpg', '.jpeg', '.png'}
        scenario_images = [f for f in scenario_path.glob(f"{scenario_name}_*.jpg")]

        results = {
            'scenario': scenario_name,
            'total_images': len(scenario_images),
            'detections': [],
            'performance_metrics': {}
        }

        for image_path in tqdm(scenario_images, desc=f"Testing {scenario_name}"):
            try:
                # Run inference
                inference_results = self.model(image_path)

                image_detections = []
                for result in inference_results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            detection = {
                                'class': int(box.cls),
                                'confidence': float(box.conf),
                                'bbox': box.xyxy[0].tolist()
                            }
                            image_detections.append(detection)

                results['detections'].extend(image_detections)

            except Exception as e:
                logger.warning(f"Error testing {image_path}: {e}")
                continue

        # Calculate performance metrics
        results['performance_metrics'] = self._calculate_scenario_metrics(results)
        return results

    def _calculate_scenario_metrics(self, results: Dict) -> Dict:
        """Calculate performance metrics for scenario"""
        detections = results['detections']

        if not detections:
            return {
                'total_detections': 0,
                'avg_confidence': 0.0,
                'class_distribution': {},
                'detection_rate': 0.0
            }

        # Calculate metrics
        confidences = [d['confidence'] for d in detections]
        classes = [d['class'] for d in detections]

        metrics = {
            'total_detections': len(detections),
            'avg_confidence': np.mean(confidences),
            'median_confidence': np.median(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'class_distribution': dict(pd.Series(classes).value_counts()),
            'detection_rate': len(detections) / results['total_images']
        }

        return metrics

    def test_all_scenarios(self) -> Dict:
        """Test model on all available scenarios"""
        # Find all scenario directories
        scenario_dirs = [d for d in self.simulation_dir.iterdir()
                        if d.is_dir() and d.name != 'results']

        all_results = {}

        for scenario_dir in scenario_dirs:
            scenario_name = scenario_dir.name
            logger.info(f"Testing scenario: {scenario_name}")

            results = self.test_on_scenario(scenario_name)
            all_results[scenario_name] = results

        # Generate comprehensive report
        report = self._generate_test_report(all_results)

        # Save results
        results_path = self.results_dir / 'simulation_test_results.json'
        with open(results_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Simulation testing completed. Results saved to: {results_path}")
        return report

    def _generate_test_report(self, all_results: Dict) -> Dict:
        """Generate comprehensive test report"""
        report = {
            'test_date': str(pd.Timestamp.now()),
            'model_path': str(self.model.ckpt_path),
            'scenarios_tested': len(all_results),
            'total_images_tested': sum(r['total_images'] for r in all_results.values()),
            'scenario_results': all_results,
            'summary_metrics': self._calculate_summary_metrics(all_results)
        }

        return report

    def _calculate_summary_metrics(self, all_results: Dict) -> Dict:
        """Calculate summary metrics across all scenarios"""
        all_detections = []
        total_images = 0

        for scenario_results in all_results.values():
            all_detections.extend(scenario_results['detections'])
            total_images += scenario_results['total_images']

        if not all_detections:
            return {
                'total_detections': 0,
                'overall_avg_confidence': 0.0,
                'overall_detection_rate': 0.0,
                'class_distribution': {}
            }

        confidences = [d['confidence'] for d in all_detections]
        classes = [d['class'] for d in all_detections]

        summary = {
            'total_detections': len(all_detections),
            'overall_avg_confidence': np.mean(confidences),
            'overall_median_confidence': np.median(confidences),
            'overall_detection_rate': len(all_detections) / total_images,
            'class_distribution': dict(pd.Series(classes).value_counts()),
            'confidence_std': np.std(confidences)
        }

        return summary

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate and test lunar surface scenarios')
    parser.add_argument('--mode', choices=['generate', 'test'], required=True,
                       help='Mode: generate scenarios or test model')
    parser.add_argument('--model-path', help='Path to trained model (required for test mode)')
    parser.add_argument('--output-dir', default='simulation/scenarios',
                       help='Output directory for generated scenarios')
    parser.add_argument('--scenarios', nargs='+',
                       choices=['crater_field', 'rocky_terrain', 'shadowed_crater',
                               'mixed_hazards', 'low_light', 'psr_simulation', 'all'],
                       default=['all'], help='Scenarios to generate/test')

    args = parser.parse_args()

    if args.mode == 'generate':
        simulator = LunarScenarioSimulator(args.output_dir)

        if 'all' in args.scenarios:
            scenarios = ['crater_field', 'rocky_terrain', 'shadowed_crater',
                        'mixed_hazards', 'low_light', 'psr_simulation']
        else:
            scenarios = args.scenarios

        for scenario in scenarios:
            simulator.generate_scenario(scenario, 10)

        print(f"Scenarios generated in: {args.output_dir}")

    elif args.mode == 'test':
        if not args.model_path:
            print("Error: --model-path required for test mode")
            return

        tester = LunarModelTester(args.model_path, args.output_dir)
        results = tester.test_all_scenarios()

        print("Testing completed. Results summary:")
        print(f"Scenarios tested: {results['scenarios_tested']}")
        print(f"Total images: {results['total_images_tested']}")
        print(f"Overall detection rate: {results['summary_metrics']['overall_detection_rate']".3f"}")
        print(f"Average confidence: {results['summary_metrics']['overall_avg_confidence']".3f"}")

if __name__ == "__main__":
    main()

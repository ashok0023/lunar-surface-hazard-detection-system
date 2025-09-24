#!/usr/bin/env python3
"""
Lunar Surface Data Augmentation
Specialized augmentation for lunar surface images, particularly for low-light conditions
and Permanently Shadowed Regions (PSR).
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import random
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class LunarAugmentation:
    """Specialized augmentation pipeline for lunar surface images"""

    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/augmented"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lunar-specific augmentation parameters
        self.lunar_transforms = {
            'low_light': self._create_low_light_transforms(),
            'shadow_enhancement': self._create_shadow_transforms(),
            'crater_emphasis': self._create_crater_transforms(),
            'noise_reduction': self._create_noise_transforms(),
            'contrast_enhancement': self._create_contrast_transforms()
        }

    def _create_low_light_transforms(self) -> A.Compose:
        """Create transforms for low-light lunar conditions"""
        return A.Compose([
            A.RandomGamma(gamma_limit=(50, 150), p=0.7),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.1),
                contrast_limit=(-0.2, 0.2),
                p=0.8
            ),
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.MotionBlur(blur_limit=3, p=0.3),
        ])

    def _create_shadow_transforms(self) -> A.Compose:
        """Create transforms to simulate and enhance shadow regions"""
        return A.Compose([
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=5,
                p=0.6
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
        ])

    def _create_crater_transforms(self) -> A.Compose:
        """Create transforms to enhance crater-like features"""
        return A.Compose([
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
        ])

    def _create_noise_transforms(self) -> A.Compose:
        """Create transforms for noise reduction and enhancement"""
        return A.Compose([
            A.GaussNoise(var_limit=(5, 25), p=0.4),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.2),
        ])

    def _create_contrast_transforms(self) -> A.Compose:
        """Create transforms for contrast enhancement in low-light"""
        return A.Compose([
            A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.6),
            A.RandomBrightnessContrast(
                brightness_limit=(0, 0.2),
                contrast_limit=(0.1, 0.3),
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3
            ),
        ])

    def augment_image(self, image_path: Path, augmentation_type: str = 'all') -> List[Path]:
        """Apply lunar-specific augmentations to an image"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return []

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented_files = []

            if augmentation_type == 'all':
                # Apply all lunar-specific augmentations
                for aug_name, transform in self.lunar_transforms.items():
                    augmented = transform(image=image)
                    aug_image = augmented['image']

                    # Save augmented image
                    output_filename = f"{image_path.stem}_{aug_name}.jpg"
                    output_path = self.output_dir / output_filename

                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_path), aug_image_bgr)
                    augmented_files.append(output_path)

            else:
                # Apply specific augmentation type
                if augmentation_type in self.lunar_transforms:
                    transform = self.lunar_transforms[augmentation_type]
                    augmented = transform(image=image)
                    aug_image = augmented['image']

                    output_filename = f"{image_path.stem}_{augmentation_type}.jpg"
                    output_path = self.output_dir / output_filename

                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_path), aug_image_bgr)
                    augmented_files.append(output_path)

            return augmented_files

        except Exception as e:
            logger.error(f"Error augmenting image {image_path}: {e}")
            return []

    def create_psr_simulation(self, image_path: Path, severity: str = 'moderate') -> Path:
        """Create Permanently Shadowed Region (PSR) simulation"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None

            # PSR simulation parameters based on severity
            psr_params = {
                'mild': {'brightness': 0.3, 'shadow_area': 0.2},
                'moderate': {'brightness': 0.2, 'shadow_area': 0.4},
                'severe': {'brightness': 0.1, 'shadow_area': 0.6}
            }

            params = psr_params.get(severity, psr_params['moderate'])

            # Create shadow mask
            height, width = image.shape[:2]
            shadow_mask = np.zeros((height, width), dtype=np.uint8)

            # Add multiple shadow regions
            num_shadows = random.randint(2, 5)
            for _ in range(num_shadows):
                x = random.randint(0, width - 100)
                y = random.randint(0, height - 100)
                w = random.randint(50, 150)
                h = random.randint(50, 150)

                cv2.rectangle(shadow_mask, (x, y), (x+w, y+h), 255, -1)

            # Apply Gaussian blur to shadow mask
            shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)

            # Create PSR effect
            psr_image = image.copy()
            psr_image = cv2.convertScaleAbs(psr_image, alpha=params['brightness'], beta=0)

            # Blend shadow regions
            mask_normalized = shadow_mask.astype(float) / 255.0
            for i in range(3):  # Apply to each channel
                psr_image[:,:,i] = (psr_image[:,:,i].astype(float) * (1 - mask_normalized * 0.8) +
                                  image[:,:,i].astype(float) * (mask_normalized * 0.2))

            psr_image = psr_image.astype(np.uint8)

            # Save PSR simulation
            output_filename = f"{image_path.stem}_PSR_{severity}.jpg"
            output_path = self.output_dir / output_filename
            cv2.imwrite(str(output_path), psr_image)

            return output_path

        except Exception as e:
            logger.error(f"Error creating PSR simulation for {image_path}: {e}")
            return None

    def batch_augment(self, augmentation_type: str = 'all', psr_severity: str = 'moderate') -> Dict[str, int]:
        """Apply augmentations to all images in input directory"""
        results = {'augmented': 0, 'psr_simulations': 0, 'errors': 0}

        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
        image_files = [f for f in self.input_dir.rglob('*')
                      if f.suffix.lower() in image_extensions]

        logger.info(f"Found {len(image_files)} images to augment")

        for image_path in tqdm(image_files, desc="Augmenting images"):
            try:
                # Apply standard augmentations
                aug_files = self.augment_image(image_path, augmentation_type)
                results['augmented'] += len(aug_files)

                # Create PSR simulation
                psr_file = self.create_psr_simulation(image_path, psr_severity)
                if psr_file:
                    results['psr_simulations'] += 1

            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results['errors'] += 1

        logger.info(f"Augmentation complete: {results}")
        return results

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Augment lunar surface images for hazard detection')
    parser.add_argument('--input-dir', default='data/raw', help='Input directory')
    parser.add_argument('--output-dir', default='data/augmented', help='Output directory')
    parser.add_argument('--augmentation-type', choices=['all', 'low_light', 'shadow_enhancement',
                                                       'crater_emphasis', 'noise_reduction',
                                                       'contrast_enhancement'],
                       default='all', help='Type of augmentation to apply')
    parser.add_argument('--psr-severity', choices=['mild', 'moderate', 'severe'],
                       default='moderate', help='PSR simulation severity')
    parser.add_argument('--batch', action='store_true', help='Process all images in input directory')

    args = parser.parse_args()

    augmentor = LunarAugmentation(args.input_dir, args.output_dir)

    if args.batch:
        results = augmentor.batch_augment(args.augmentation_type, args.psr_severity)
        print(f"Batch augmentation complete: {results}")
    else:
        # Process single image (for testing)
        print("Please use --batch flag to process all images")

if __name__ == "__main__":
    main()

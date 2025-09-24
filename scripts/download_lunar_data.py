#!/usr/bin/env python3
"""
Lunar Surface Image Downloader
Downloads lunar surface images from various sources for hazard detection training.
"""

import os
import requests
import json
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LunarDataDownloader:
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # NASA Lunar Reconnaissance Orbiter (LRO) API endpoints
        self.lro_base_url = "https://api.nasa.gov/lro"

        # Simulated lunar surface data sources
        self.data_sources = {
            'nasa_lro': {
                'base_url': 'https://api.nasa.gov/lro/browse',
                'api_key_required': True,
                'description': 'NASA Lunar Reconnaissance Orbiter data'
            },
            'usgs_moon': {
                'base_url': 'https://astrogeology.usgs.gov/api/moon',
                'api_key_required': False,
                'description': 'USGS Astrogeology lunar data'
            }
        }

    def download_from_nasa_lro(self, api_key: str, max_images: int = 100) -> List[str]:
        """Download images from NASA LRO API"""
        downloaded_files = []

        try:
            # Example: Search for lunar surface images
            search_url = f"{self.lro_base_url}/search"
            params = {
                'q': 'lunar surface hazard',
                'media_type': 'image',
                'api_key': api_key
            }

            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Process results (simplified for this example)
            for item in data.get('collection', {}).get('items', [])[:max_images]:
                if 'links' in item:
                    image_url = item['links'][0]['href']
                    filename = self.output_dir / f"lro_{len(downloaded_files)}.jpg"

                    if self._download_image(image_url, filename):
                        downloaded_files.append(str(filename))

            logger.info(f"Downloaded {len(downloaded_files)} images from NASA LRO")

        except Exception as e:
            logger.error(f"Error downloading from NASA LRO: {e}")

        return downloaded_files

    def download_from_usgs(self, max_images: int = 50) -> List[str]:
        """Download images from USGS Astrogeology"""
        downloaded_files = []

        try:
            # USGS Moon data API (simplified example)
            api_url = "https://astrogeology.usgs.gov/api/moon/lunar_surface"

            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()

            # Process lunar surface images
            for item in data.get('images', [])[:max_images]:
                image_url = item.get('url')
                if image_url:
                    filename = self.output_dir / f"usgs_{len(downloaded_files)}.jpg"

                    if self._download_image(image_url, filename):
                        downloaded_files.append(str(filename))

            logger.info(f"Downloaded {len(downloaded_files)} images from USGS")

        except Exception as e:
            logger.error(f"Error downloading from USGS: {e}")

        return downloaded_files

    def _download_image(self, url: str, filename: Path) -> bool:
        """Download a single image"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded: {filename}")
            return True

        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False

    def download_sample_dataset(self) -> List[str]:
        """Download a sample lunar dataset for testing"""
        # Create sample lunar-like images using synthetic data
        sample_dir = self.output_dir / "sample"
        sample_dir.mkdir(exist_ok=True)

        # Generate some sample images for demonstration
        sample_files = []

        # Create a few sample lunar-like images
        for i in range(5):
            # Create a simple gradient image to simulate lunar surface
            height, width = 480, 640
            image = np.full((height, width, 3), 128, dtype=np.uint8)

            # Add some texture
            noise = np.random.normal(0, 15, (height, width, 3))
            image = np.clip(image + noise.astype(np.int16), 0, 255).astype(np.uint8)

            # Add some crater-like features
            center_x, center_y = width // 2, height // 2
            y, x = np.ogrid[:height, :width]
            crater_mask = np.sqrt((x - center_x)**2 + (y - center_y)**2) <= 50
            image[crater_mask] = image[crater_mask] * 0.7  # Darken crater area

            # Save sample image
            sample_path = sample_dir / f"sample_lunar_{i+1}.jpg"
            cv2.imwrite(str(sample_path), image)
            sample_files.append(str(sample_path))

        logger.info(f"Generated {len(sample_files)} sample lunar images")
        return sample_files

    def download_all_sources(self, nasa_api_key: Optional[str] = None) -> Dict[str, List[str]]:
        """Download from all available sources"""
        results = {}

        if nasa_api_key:
            results['nasa_lro'] = self.download_from_nasa_lro(nasa_api_key)
        else:
            logger.warning("NASA API key not provided, skipping LRO download")

        results['usgs'] = self.download_from_usgs()
        results['sample'] = self.download_sample_dataset()

        return results

def main():
    parser = argparse.ArgumentParser(description='Download lunar surface images for hazard detection')
    parser.add_argument('--output-dir', default='data/raw', help='Output directory')
    parser.add_argument('--nasa-api-key', help='NASA API key for LRO data')
    parser.add_argument('--max-images', type=int, default=100, help='Maximum images per source')
    parser.add_argument('--sources', nargs='+', choices=['nasa', 'usgs', 'sample', 'all'],
                       default=['all'], help='Data sources to download from')
    parser.add_argument('--batch', action='store_true', help='Process all sources (same as default behavior)')

    args = parser.parse_args()

    downloader = LunarDataDownloader(args.output_dir)

    if 'all' in args.sources:
        sources = ['nasa', 'usgs', 'sample']
    else:
        sources = args.sources

    total_downloaded = 0

    for source in sources:
        if source == 'nasa' and args.nasa_api_key:
            files = downloader.download_from_nasa_lro(args.nasa_api_key, args.max_images)
            total_downloaded += len(files)
        elif source == 'usgs':
            files = downloader.download_from_usgs(args.max_images)
            total_downloaded += len(files)
        elif source == 'sample':
            files = downloader.download_sample_dataset()
            total_downloaded += len(files)

    logger.info(f"Total images downloaded: {total_downloaded}")

if __name__ == "__main__":
    main()

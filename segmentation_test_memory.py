import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import math
import torchvision
import sys
import time

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run SAM model on an image.")
    parser.add_argument("image_path", type=str, help="Path to the image.")
    parser.add_argument("device", type=str, choices=["cpu", "cuda"], help="Device to run the model on.")
    parser.add_argument("seg_strength", type=str, choices=["weak", "strong"], help="Segmentation strength.")

    args = parser.parse_args()

    # Read and process the image
    image = cv2.imread(args.image_path)
    if image is None:
        raise IOError("Unable to read the image from the path: {}".format(args.image_path))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=args.device)

    if args.seg_strength == 'weak':
        mask_generator = SamAutomaticMaskGenerator(sam)

    if args.seg_strength == 'strong':
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=64,
            pred_iou_thresh=0.60,
            stability_score_thresh=0.80,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )

    # Reset peak memory stats
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Measure time before the operation
    start_time = time.time()

    # Run the operation
    masks = mask_generator.generate(image)

    # Measure time after the operation
    end_time = time.time()

    # Calculate and print the duration
    duration = end_time - start_time
    print("Time taken for mask generation: {} seconds".format(duration))

    # Check peak memory usage
    if args.device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated()
        peak_memory_MB = peak_memory / (1024 * 1024)
        print("Peak memory used for operation: {} MB".format(peak_memory_MB))

if __name__ == "__main__":
    main()

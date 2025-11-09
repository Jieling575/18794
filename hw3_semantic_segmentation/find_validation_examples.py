"""
Find validation set examples for specific categories (for Question 5)
Since Problem 1 uses training set, we need to find similar examples in validation set
"""

import os
import numpy as np
from PIL import Image
from datasets import VOCSegmentation
import argparse

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def find_validation_images_with_classes(data_root, class_names, max_images=5):
    """
    Find validation images that contain the specified classes.
    
    Args:
        data_root: Path to dataset
        class_names: List of class names to find (e.g., ['person', 'car', 'dog'])
        max_images: Maximum number of images to find per class
    
    Returns:
        Dictionary mapping class names to list of (image_idx, image_name) tuples
    """
    dataset = VOCSegmentation(root=data_root, image_set='val', transform=None)
    
    # Convert class names to class IDs
    target_class_ids = []
    for name in class_names:
        if name in VOC_CLASSES:
            class_id = VOC_CLASSES.index(name) + 1  # +1 because background is 0
            target_class_ids.append((class_id, name))
        else:
            print(f"Warning: '{name}' not found in VOC classes")
    
    print(f"\nSearching for {len(target_class_ids)} classes in validation set...")
    print(f"Target classes: {[name for _, name in target_class_ids]}\n")
    
    # Find images for each class
    class_to_images = {name: [] for _, name in target_class_ids}
    
    for idx in range(len(dataset)):
        # Check if we've found enough images for all classes
        if all(len(imgs) >= max_images for imgs in class_to_images.values()):
            break
        
        # Load mask
        mask = np.array(Image.open(dataset.masks[idx]))
        unique_classes = np.unique(mask)
        
        # Check each target class
        for class_id, class_name in target_class_ids:
            if class_id in unique_classes and len(class_to_images[class_name]) < max_images:
                # Get image name from path
                img_path = dataset.images[idx]
                img_name = os.path.basename(img_path).replace('.jpg', '')
                
                class_to_images[class_name].append((idx, img_name))
                print(f"  {class_name:15s}: Found at index {idx:4d} (image: {img_name})")
        
        if (idx + 1) % 100 == 0:
            found = sum(len(imgs) for imgs in class_to_images.values())
            print(f"  ... processed {idx + 1} images, found {found} total examples")
    
    return class_to_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./datasets/data')
    parser.add_argument('--categories', type=str, nargs='+', 
                        default=['person', 'car', 'dog', 'bicycle', 'sofa'],
                        help='List of categories to find (pick 5 from Problem 1)')
    parser.add_argument('--num_images', type=int, default=1,
                        help='Number of images to find per category')
    args = parser.parse_args()
    
    print("="*70)
    print("Finding Validation Images for Question 5")
    print("="*70)
    print(f"\nSearching for categories: {args.categories}")
    
    results = find_validation_images_with_classes(args.data_root, args.categories, args.num_images)
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Collect all unique image names
    all_images = []
    for class_name, images in results.items():
        print(f"\n{class_name}:")
        for idx, img_name in images:
            print(f"  Index {idx}: {img_name}")
            if img_name not in all_images:
                all_images.append(img_name)
    
    # If we want exactly 5 images total (one per category)
    if args.num_images == 1 and len(all_images) <= 5:
        print("\n" + "="*70)
        print("COMMAND FOR QUESTION 5")
        print("="*70)
        print("\nUse these 5 images:\n")
        print("python compare_models.py \\")
        print("  --model1_ckpt checkpoints/best_deeplabv3plus_resnet50_VOC_os16.pth \\")
        print("  --model2_ckpt checkpoints/best_deeplabv3plus_resnet50_VOC_os16_focal.pth \\")
        print("  --model1_name \"Cross Entropy\" \\")
        print("  --model2_name \"Focal Loss\" \\")
        print(f"  --image_names {' '.join(all_images)} \\")
        print("  --output_dir comparison_results")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()


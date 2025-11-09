"""
Helper script to identify which images were used in Problem 1
"""

import os
import argparse


def find_images_from_directory(directory):
    """Find image names from a directory containing Problem 1 visualizations"""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    
    image_names = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg')):
            # Extract the image ID from filename
            # Typical format: something_2007_000032_something.png
            parts = filename.split('_')
            for i, part in enumerate(parts):
                if part.startswith('200') or part.startswith('201'):  # Year prefix
                    if i + 1 < len(parts):
                        img_id = f"{part}_{parts[i+1].split('.')[0]}"
                        if img_id not in image_names:
                            image_names.append(img_id)
                            break
    
    return image_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualization_dir", type=str, default=None,
                        help="Directory containing Problem 1 visualizations")
    args = parser.parse_args()
    
    print("="*60)
    print("Finding Images from Problem 1")
    print("="*60)
    
    if args.visualization_dir:
        print(f"\nSearching in directory: {args.visualization_dir}")
        image_names = find_images_from_directory(args.visualization_dir)
        
        if image_names:
            print(f"\nFound {len(image_names)} images:")
            for name in image_names:
                print(f"  - {name}")
            
            print("\n" + "="*60)
            print("Command to use these images:")
            print("="*60)
            print(f"\npython compare_models.py \\")
            print(f"  --model1_ckpt checkpoints/best_*.pth \\")
            print(f"  --model2_ckpt checkpoints/best_*_focal.pth \\")
            print(f"  --model1_name \"Cross Entropy\" \\")
            print(f"  --model2_name \"Focal Loss\" \\")
            print(f"  --image_names {' '.join(image_names)}")
        else:
            print("\nNo images found in the specified directory.")
    else:
        print("\nPlease provide options:")
        print("\n1. If you have Problem 1 visualizations in a directory:")
        print("   python find_problem1_images.py --visualization_dir path/to/problem1/")
        print("\n2. Or manually list the 5 image IDs you used:")
        print("   Example: 2007_000032, 2008_000015, 2009_001234, 2010_000456, 2011_000789")
        print("\n3. Then use them with compare_models.py:")
        print("   python compare_models.py \\")
        print("     --image_names 2007_000032 2008_000015 2009_001234 2010_000456 2011_000789 \\")
        print("     --model1_ckpt ... --model2_ckpt ...")


if __name__ == '__main__':
    main()


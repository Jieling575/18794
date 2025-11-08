import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datasets import VOCSegmentation
from collections import defaultdict

# VOC class names (excluding background which is class 0)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def find_images_per_class(dataset, num_classes=21):
    """
    Find one image for each object class in the dataset.
    Returns a dict mapping class_id to image_index.
    """
    class_to_image = {}
    
    print("Scanning dataset to find representative images for each class...")
    for idx in range(len(dataset)):
        if len(class_to_image) == num_classes - 1:  # -1 to exclude background
            break
            
        # Load mask without transforms
        mask_path = dataset.masks[idx]
        mask = np.array(Image.open(mask_path))
        
        # Find unique classes in this image
        unique_classes = np.unique(mask)
        
        # For each class found (excluding background=0 and boundary=255)
        for class_id in unique_classes:
            if class_id == 0 or class_id == 255:  # Skip background and boundary
                continue
            if class_id not in class_to_image:
                class_to_image[class_id] = idx
                print(f"Found {VOC_CLASSES[class_id-1]} (class {class_id}) in image {idx}")
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} images, found {len(class_to_image)} classes so far")
    
    return class_to_image

def decode_segmap(mask, nc=21):
    """
    Decode segmentation mask to RGB for visualization using VOC colormap.
    """
    label_colors = np.array([
        [0, 0, 0],  # 0=background
        # 20 object classes
        [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
        [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0],
        [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
    ])
    
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    
    for l in range(nc):
        idx = mask == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def visualize_all_categories(data_root='./datasets/data', save_path='category_visualization.png'):
    """
    Create a visualization of one image per category with ground truth masks.
    Layout: 10 pairs per row, 2 rows for 20 classes.
    Each pair shows: original image | ground truth mask
    """
    # Load dataset without transforms
    dataset = VOCSegmentation(root=data_root, image_set='train', transform=None)
    
    # Find one image per class
    class_to_image = find_images_per_class(dataset)
    
    # Sort by class ID to maintain order
    sorted_classes = sorted(class_to_image.keys())
    
    if len(sorted_classes) < 20:
        print(f"\nWarning: Only found {len(sorted_classes)} out of 20 classes")
        print(f"Missing classes: {[VOC_CLASSES[i] for i in range(20) if (i+1) not in sorted_classes]}")
    
    # Create figure: 2 rows, 10 pairs per row = 2 rows × 20 columns (each pair is 2 columns)
    pairs_per_row = 10
    num_rows = int(np.ceil(len(sorted_classes) / pairs_per_row))
    
    fig, axes = plt.subplots(num_rows, pairs_per_row * 2, figsize=(40, 8 * num_rows))
    fig.suptitle('VOC Dataset: One Image Per Category (Image | Ground Truth Mask)', 
                 fontsize=24, y=0.995)
    
    # Flatten axes for easier indexing
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, class_id in enumerate(sorted_classes):
        row = idx // pairs_per_row
        col_base = (idx % pairs_per_row) * 2
        
        image_idx = class_to_image[class_id]
        class_name = VOC_CLASSES[class_id - 1]
        
        # Load image and mask
        img = Image.open(dataset.images[image_idx]).convert('RGB')
        mask = np.array(Image.open(dataset.masks[image_idx]))
        
        # Decode mask to RGB
        mask_rgb = decode_segmap(mask)
        
        # Plot original image
        axes[row, col_base].imshow(img)
        axes[row, col_base].axis('off')
        axes[row, col_base].set_title(f'{class_name}', fontsize=12, pad=5)
        
        # Plot mask
        axes[row, col_base + 1].imshow(mask_rgb)
        axes[row, col_base + 1].axis('off')
        axes[row, col_base + 1].set_title(f'{class_name} (mask)', fontsize=12, pad=5)
    
    # Hide any unused subplots
    total_pairs = len(sorted_classes)
    for idx in range(total_pairs, num_rows * pairs_per_row):
        row = idx // pairs_per_row
        col_base = (idx % pairs_per_row) * 2
        axes[row, col_base].axis('off')
        axes[row, col_base + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()
    
    return sorted_classes

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize one image per VOC category')
    parser.add_argument('--data_root', type=str, default='./datasets/data',
                        help='Path to dataset root')
    parser.add_argument('--save_path', type=str, default='category_visualization.png',
                        help='Path to save the output image')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VOC Dataset Category Visualization")
    print("="*60)
    
    classes_found = visualize_all_categories(args.data_root, args.save_path)
    
    print("\n" + "="*60)
    print(f"Successfully visualized {len(classes_found)} categories!")
    print("="*60)


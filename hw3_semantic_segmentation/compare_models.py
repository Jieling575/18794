"""
Script to compare predictions from two trained models side-by-side
For Question 5: Visualize RGB Image | Ground Truth | Model1 | Model2
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse

import network
from datasets import VOCSegmentation
from utils import ext_transforms as et


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--model1_ckpt", type=str, required=True,
                        help="checkpoint path for model 1 (e.g., Cross Entropy)")
    parser.add_argument("--model2_ckpt", type=str, required=True,
                        help="checkpoint path for model 2 (e.g., Focal Loss)")
    parser.add_argument("--model1_name", type=str, default="Cross Entropy",
                        help="name for model 1")
    parser.add_argument("--model2_name", type=str, default="Focal Loss",
                        help="name for model 2")
    parser.add_argument("--num_images", type=int, default=5,
                        help="number of images to visualize")
    parser.add_argument("--output_dir", type=str, default="comparison_results",
                        help="output directory for visualizations")
    parser.add_argument("--image_indices", type=int, nargs='+', default=None,
                        help="specific image indices to visualize (0-indexed)")
    parser.add_argument("--image_names", type=str, nargs='+', default=None,
                        help="specific image names to visualize (e.g., 2007_000032)")
    return parser


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model
    model = network.modeling.deeplabv3plus_resnet50(num_classes=21, output_stride=16)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DataParallel state dict
    state_dict = checkpoint["model_state"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"  Best mIoU: {checkpoint.get('best_score', 'N/A')}")
    
    return model, checkpoint.get('best_score', None)


def predict(model, image_tensor, device):
    """Get prediction from model"""
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        pred = output.max(1)[1].cpu().numpy()[0]
    return pred


def visualize_comparison(rgb_image, gt_mask, pred1, pred2, 
                         model1_name, model2_name, save_path):
    """Create side-by-side comparison visualization"""
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # RGB Image
    axes[0].imshow(rgb_image)
    axes[0].set_title('RGB Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(gt_mask)
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Model 1 Prediction
    axes[2].imshow(pred1)
    axes[2].set_title(f'{model1_name}\nPrediction', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Model 2 Prediction
    axes[3].imshow(pred2)
    axes[3].set_title(f'{model2_name}\nPrediction', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison to {save_path}")


def main():
    opts = get_argparser().parse_args()
    
    # Create output directory
    os.makedirs(opts.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load both models
    print("\n" + "="*60)
    print("Loading Models")
    print("="*60)
    model1, best_score1 = load_model(opts.model1_ckpt, device)
    model2, best_score2 = load_model(opts.model2_ckpt, device)
    
    # Setup dataset (validation set)
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)
    
    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    
    val_dst = VOCSegmentation(root=opts.data_root, image_set='val', 
                              transform=val_transform)
    
    print(f"Validation set: {len(val_dst)} images")
    
    # Determine which images to visualize
    if opts.image_names is not None:
        # Find indices by image names
        print(f"\nSearching for images by name: {opts.image_names}")
        indices = []
        
        # Get image file names from dataset
        val_dst_raw = VOCSegmentation(root=opts.data_root, image_set='val', transform=None)
        for img_name in opts.image_names:
            # Image files are named like: 2007_000032.jpg
            if not img_name.endswith('.jpg'):
                img_name = img_name + '.jpg'
            
            # Find the index of this image in the dataset
            found = False
            for idx in range(len(val_dst_raw)):
                dataset_img_path = val_dst_raw.images[idx]
                if img_name in dataset_img_path:
                    indices.append(idx)
                    print(f"  Found {img_name} at index {idx}")
                    found = True
                    break
            
            if not found:
                print(f"  WARNING: Image {img_name} not found in validation set!")
        
        if len(indices) == 0:
            print("ERROR: No images found! Using default selection.")
            step = len(val_dst) // opts.num_images
            indices = [i * step for i in range(opts.num_images)]
    
    elif opts.image_indices is not None:
        indices = opts.image_indices
    else:
        # Select diverse images (spread across dataset)
        step = len(val_dst) // opts.num_images
        indices = [i * step for i in range(opts.num_images)]
    
    print(f"\nVisualizing images at indices: {indices}")
    
    # Generate comparisons
    print("\n" + "="*60)
    print("Generating Comparisons")
    print("="*60)
    
    for idx, img_idx in enumerate(indices):
        print(f"\nProcessing image {idx+1}/{len(indices)} (index {img_idx})")
        
        # Get original RGB image and ground truth
        # Load without transform for visualization
        val_dst_raw = VOCSegmentation(root=opts.data_root, image_set='val', 
                                      transform=None)
        rgb_image, gt_mask = val_dst_raw[img_idx]
        rgb_image_np = np.array(rgb_image)
        gt_mask_np = np.array(gt_mask)
        
        # Get transformed image for model input
        image_tensor, _ = val_dst[img_idx]
        
        # Get predictions from both models
        pred1 = predict(model1, image_tensor, device)
        pred2 = predict(model2, image_tensor, device)
        
        # Decode masks to RGB for visualization
        gt_rgb = val_dst.decode_target(gt_mask_np)
        pred1_rgb = val_dst.decode_target(pred1)
        pred2_rgb = val_dst.decode_target(pred2)
        
        # Create comparison visualization
        save_path = os.path.join(opts.output_dir, f'comparison_{idx+1}.png')
        visualize_comparison(rgb_image_np, gt_rgb, pred1_rgb, pred2_rgb,
                           opts.model1_name, opts.model2_name, save_path)
    
    # Generate summary report
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    print(f"\n{opts.model1_name}:")
    print(f"  Checkpoint: {opts.model1_ckpt}")
    print(f"  Best mIoU: {best_score1*100:.2f}%" if best_score1 else "  Best mIoU: N/A")
    
    print(f"\n{opts.model2_name}:")
    print(f"  Checkpoint: {opts.model2_ckpt}")
    print(f"  Best mIoU: {best_score2*100:.2f}%" if best_score2 else "  Best mIoU: N/A")
    
    print(f"\nGenerated {len(indices)} comparison images in: {opts.output_dir}/")
    print("="*60)
    
    # Create a LaTeX-ready table
    latex_table = f"""
\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{|l|c|c|}}
\\hline
\\textbf{{Model}} & \\textbf{{mIoU (\\%)}} & \\textbf{{Accuracy (\\%)}} \\\\
\\hline
{opts.model1_name} & {best_score1*100:.2f} & - \\\\
{opts.model2_name} & {best_score2*100:.2f} & - \\\\
\\hline
\\end{{tabular}}
\\caption{{Performance comparison of trained models on Pascal VOC validation set.}}
\\end{{table}}
"""
    
    # Save LaTeX table
    with open(os.path.join(opts.output_dir, 'performance_table.tex'), 'w') as f:
        f.write(latex_table)
    
    print("\nLaTeX table saved to:", os.path.join(opts.output_dir, 'performance_table.tex'))


if __name__ == '__main__':
    main()


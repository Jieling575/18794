"""
Evaluate a trained model and report detailed metrics
"""

import torch
import torch.nn as nn
from torch.utils import data
import argparse
import numpy as np

import network
from datasets import VOCSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="checkpoint path")
    parser.add_argument("--model_name", type=str, default="Model",
                        help="name of the model")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="batch size for evaluation")
    return parser


def validate(model, loader, device, metrics):
    """Validate model and compute metrics"""
    metrics.reset()
    model.eval()
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            
            metrics.update(targets, preds)
    
    return metrics.get_results()


def main():
    opts = get_argparser().parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model from: {opts.ckpt}")
    model = network.modeling.deeplabv3plus_resnet50(num_classes=21, output_stride=16)
    
    checkpoint = torch.load(opts.ckpt, map_location=device)
    
    # Handle DataParallel state dict
    state_dict = checkpoint["model_state"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Checkpoint best score: {checkpoint.get('best_score', 'N/A')}")
    
    # Setup dataset
    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    
    val_dst = VOCSegmentation(root=opts.data_root, image_set='val', 
                              transform=val_transform)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size, 
                                 shuffle=False, num_workers=2)
    
    print(f"Validation set: {len(val_dst)} images")
    
    # Setup metrics
    metrics = StreamSegMetrics(21)
    
    # Evaluate
    print("\nEvaluating...")
    scores = validate(model, val_loader, device, metrics)
    
    # Print results
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS - {opts.model_name}")
    print("="*60)
    print(f"Overall Accuracy: {scores['Overall Acc']:.4f} ({scores['Overall Acc']*100:.2f}%)")
    print(f"Mean Accuracy:    {scores['Mean Acc']:.4f} ({scores['Mean Acc']*100:.2f}%)")
    print(f"FreqW Accuracy:   {scores['FreqW Acc']:.4f} ({scores['FreqW Acc']*100:.2f}%)")
    print(f"Mean IoU:         {scores['Mean IoU']:.4f} ({scores['Mean IoU']*100:.2f}%)")
    print("="*60)
    
    # Print per-class IoU
    print("\nPer-Class IoU:")
    print("-"*60)
    class_names = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    class_iou = scores['Class IoU']
    for idx, (name, iou) in enumerate(zip(class_names, class_iou)):
        print(f"{name:15s}: {iou:.4f} ({iou*100:.2f}%)")
    print("="*60)
    
    # Save results to file
    output_file = f"{opts.model_name.replace(' ', '_')}_results.txt"
    with open(output_file, 'w') as f:
        f.write(f"Evaluation Results - {opts.model_name}\n")
        f.write("="*60 + "\n")
        f.write(f"Overall Accuracy: {scores['Overall Acc']:.4f} ({scores['Overall Acc']*100:.2f}%)\n")
        f.write(f"Mean Accuracy:    {scores['Mean Acc']:.4f} ({scores['Mean Acc']*100:.2f}%)\n")
        f.write(f"FreqW Accuracy:   {scores['FreqW Acc']:.4f} ({scores['FreqW Acc']*100:.2f}%)\n")
        f.write(f"Mean IoU:         {scores['Mean IoU']:.4f} ({scores['Mean IoU']*100:.2f}%)\n")
        f.write("="*60 + "\n\n")
        f.write("Per-Class IoU:\n")
        f.write("-"*60 + "\n")
        for name, iou in zip(class_names, class_iou):
            f.write(f"{name:15s}: {iou:.4f} ({iou*100:.2f}%)\n")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()


# Training Guide for DeepLabV3+ on Pascal VOC

## Setup Complete ✓

All code implementations are complete:
- ✓ Dataset loading (VOCSegmentation)
- ✓ ASPP modules (ASPPConv, ASPPPooling, ASPP)
- ✓ DeepLab heads (DeepLabHead, DeepLabHeadV3Plus)
- ✓ Optimizer with scaled learning rates (backbone: 0.1x, classifier: 1.0x)
- ✓ Step learning rate scheduler (decay by 0.1 every 1000 iterations)
- ✓ Training loop with mIoU tracking

## Training Command

### Option 1: Train with Default Settings (Recommended)

```bash
python main.py --model deeplabv3plus_resnet50 --output_stride 16 --batch_size 8 --lr 0.01 --lr_policy step --step_size 1000 --total_itrs 5000 --gpu_id 0
```

### Option 2: Train with Output Stride 8 (Higher accuracy, more memory)

```bash
python main.py --model deeplabv3plus_resnet50 --output_stride 8 --batch_size 4 --lr 0.01 --lr_policy step --step_size 1000 --total_itrs 5000 --gpu_id 0
```

## Training Parameters Explained

- `--model deeplabv3plus_resnet50`: Use DeepLabV3+ with ResNet50 backbone
- `--output_stride 16`: Controls feature map resolution (16 = 1/16 of input size)
- `--batch_size 8`: Number of images per batch (adjust based on GPU memory)
- `--lr 0.01`: Base learning rate (backbone will use 0.001, classifier will use 0.01)
- `--lr_policy step`: Use step learning rate decay
- `--step_size 1000`: Decay learning rate every 1000 iterations
- `--total_itrs 5000`: Train for 5000 iterations total
- `--gpu_id 0`: GPU device ID

## Expected Training Behavior

### Training Progress
- **Iterations**: 5000 total
- **Epochs**: ~3-4 epochs (depends on batch size)
- **Time**: ~1-2 hours on modern GPU
- **Memory**: ~6-7 GB GPU memory with batch_size=8

### Learning Rate Schedule
| Iteration | Backbone LR | Classifier LR |
|-----------|-------------|---------------|
| 0-999     | 0.001       | 0.01          |
| 1000-1999 | 0.0001      | 0.001         |
| 2000-2999 | 0.00001     | 0.0001        |
| 3000-3999 | 0.000001    | 0.00001       |
| 4000-5000 | 0.0000001   | 0.000001      |

### Expected mIoU Progress
- **Epoch 1**: ~45-55% mIoU
- **Epoch 2**: ~60-65% mIoU
- **Epoch 3**: ~65-67% mIoU (target: ≥65%)

## Monitoring Training

### During Training
The terminal will show:
```
Epoch 1, Itrs 10/5000, Loss=1.234567
...
validation...
Overall Acc: 0.8500, Mean Acc: 0.7800, FreqW Acc: 0.7900, Mean IoU: 0.6234
Epoch 1: mIoU = 0.6234
new best mIOU: 0.6234
Model saved as checkpoints/best_deeplabv3plus_resnet50_VOC_os16.pth
```

### After Training
1. **View mIoU plot**:
```bash
python plot_miou.py
```

2. **Check saved files**:
- `checkpoints/best_deeplabv3plus_resnet50_VOC_os16.pth` - Best model
- `checkpoints/latest_deeplabv3plus_resnet50_VOC_os16.pth` - Latest model
- `checkpoints/mIoU_history.npy` - mIoU per epoch data
- `miou_plot.png` - mIoU visualization plot

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size:
```bash
python main.py --model deeplabv3plus_resnet50 --batch_size 4 ...
```

### Training Too Slow
Use smaller crop size:
```bash
python main.py --model deeplabv3plus_resnet50 --crop_size 400 ...
```

### Resume Training
```bash
python main.py --ckpt checkpoints/latest_deeplabv3plus_resnet50_VOC_os16.pth --continue_training ...
```

## Testing Only

To test a trained model without training:
```bash
python main.py --model deeplabv3plus_resnet50 --ckpt checkpoints/best_deeplabv3plus_resnet50_VOC_os16.pth --test_only
```

## For Your Report

1. **Include the mIoU plot**: `miou_plot.png`
2. **Report best mIoU**: Should be ≥65%
3. **Training settings**:
   - Model: DeepLabV3+ with ResNet50 (ImageNet pretrained)
   - Loss: CrossEntropyLoss
   - Optimizer: SGD with momentum=0.9, weight_decay=1e-4
   - Learning rate: 0.01 (classifier), 0.001 (backbone)
   - Scheduler: StepLR (decay by 0.1 every 1000 iterations)
   - Batch size: 8
   - Total iterations: 5000
   - Dataset: Pascal VOC 2012 train/val split

## Good Luck! 🚀



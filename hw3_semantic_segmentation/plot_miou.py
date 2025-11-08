import numpy as np
import matplotlib.pyplot as plt
import os

def plot_miou_history(history_file='checkpoints/mIoU_history.npy', save_path='miou_plot.png'):
    """
    Plot the mIoU per epoch from training history.
    
    Args:
        history_file: Path to the saved mIoU history (.npy file)
        save_path: Path to save the plot
    """
    if not os.path.exists(history_file):
        print(f"Error: History file not found at {history_file}")
        print("Please train the model first to generate the history.")
        return
    
    # Load mIoU history
    miou_history = np.load(history_file)
    epochs = np.arange(1, len(miou_history) + 1)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, miou_history * 100, 'b-', linewidth=2, marker='o', markersize=6)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Mean IoU (%)', fontsize=14)
    plt.title('Validation mIoU per Epoch - DeepLabV3+ on Pascal VOC', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Add best mIoU annotation
    best_miou = np.max(miou_history) * 100
    best_epoch = np.argmax(miou_history) + 1
    plt.axhline(y=best_miou, color='r', linestyle='--', alpha=0.5, label=f'Best mIoU: {best_miou:.2f}%')
    plt.plot(best_epoch, best_miou, 'r*', markersize=15)
    
    # Add text with statistics
    final_miou = miou_history[-1] * 100
    textstr = f'Best mIoU: {best_miou:.2f}% (Epoch {best_epoch})\n'
    textstr += f'Final mIoU: {final_miou:.2f}% (Epoch {len(miou_history)})'
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Plot saved to: {save_path}")
    print(f"{'='*60}")
    print(f"Training Summary:")
    print(f"  Total Epochs: {len(miou_history)}")
    print(f"  Best mIoU: {best_miou:.2f}% (Epoch {best_epoch})")
    print(f"  Final mIoU: {final_miou:.2f}% (Epoch {len(miou_history)})")
    print(f"  Initial mIoU: {miou_history[0]*100:.2f}% (Epoch 1)")
    print(f"  Improvement: {final_miou - miou_history[0]*100:.2f}%")
    print(f"{'='*60}\n")
    
    # Also print epoch-by-epoch data
    print("\nEpoch-by-Epoch mIoU:")
    print(f"{'Epoch':<10} {'mIoU (%)':<15}")
    print("-" * 25)
    for epoch, miou in enumerate(miou_history, 1):
        marker = " ← Best" if epoch == best_epoch else ""
        print(f"{epoch:<10} {miou*100:<15.2f}{marker}")
    
    plt.show()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot mIoU history from training')
    parser.add_argument('--history_file', type=str, default='checkpoints/mIoU_history.npy',
                        help='Path to the mIoU history file')
    parser.add_argument('--save_path', type=str, default='miou_plot.png',
                        help='Path to save the plot')
    
    args = parser.parse_args()
    
    plot_miou_history(args.history_file, args.save_path)


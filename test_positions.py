"""Test script to verify shapes appear at different positions."""
import matplotlib.pyplot as plt
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from grokking_mnist.config import GrokConfig
    from grokking_mnist.data import load_geometric_shapes
else:
    from .config import GrokConfig
    from .data import load_geometric_shapes


def test_shape_positions():
    """Generate shapes and visualize to verify position variation."""
    config = GrokConfig(seed=42)
    train_set, _ = load_geometric_shapes(config, num_samples_per_shape=20)
    
    shape_names = ['Lines', 'Circles', 'Triangles', 'Squares']
    
    fig, axes = plt.subplots(4, 10, figsize=(20, 8))
    fig.suptitle('Position Variation Test - Same Shape Type at Different Locations', 
                 fontsize=14, fontweight='bold')
    
    for shape_idx, shape_name in enumerate(shape_names):
        indices = [i for i, label in enumerate(train_set.labels) if label == shape_idx]
        
        for col in range(10):
            sample_idx = indices[col]
            img = train_set.data[sample_idx][0]
            
            axes[shape_idx, col].imshow(img, cmap='gray')
            axes[shape_idx, col].axis('off')
            
            if col == 0:
                axes[shape_idx, col].set_ylabel(shape_name, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs('./results', exist_ok=True)
    output_path = './results/position_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Position test visualization saved to: {output_path}")
    print("If shapes appear at different locations in each column, position variation is working!")
    plt.show()


if __name__ == "__main__":
    test_shape_positions()

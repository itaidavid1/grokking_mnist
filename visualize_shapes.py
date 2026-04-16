"""Visualize the geometric shapes dataset with variations."""
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


def visualize_shapes():
    """Generate and display sample geometric shapes."""
    config = GrokConfig(seed=42)
    train_set, _ = load_geometric_shapes(config.seed, num_samples_per_shape=50)
    
    shape_names = ['Lines', 'Circles', 'Triangles', 'Squares']
    
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle('Geometric Shapes Dataset - Sample Variations\n'
                 '(Different sizes, orientations, and sharpness levels)', 
                 fontsize=14, fontweight='bold')
    
    for shape_idx, shape_name in enumerate(shape_names):
        indices = [i for i, label in enumerate(train_set.labels) if label == shape_idx]
        
        for col in range(8):
            sample_idx = indices[col]
            img = train_set.data[sample_idx][0]
            
            axes[shape_idx, col].imshow(img, cmap='gray')
            axes[shape_idx, col].axis('off')
            
            if col == 0:
                axes[shape_idx, col].set_ylabel(shape_name, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs('./results', exist_ok=True)
    output_path = './results/geometric_shapes_samples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    visualize_shapes()

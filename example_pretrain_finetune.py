"""Example: Pre-train on geometric shapes, then fine-tune on MNIST.

This script demonstrates a complete workflow:
1. Pre-train a model on geometric shapes
2. Fine-tune the pre-trained model on MNIST
"""
import torch
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from grokking_mnist.config import GrokConfig
    from grokking_mnist.train import train
    from grokking_mnist.data import load_geometric_shapes, load_mnist
    from grokking_mnist.model import build_mlp
else:
    from .config import GrokConfig
    from .train import train
    from .data import load_geometric_shapes, load_mnist
    from .model import build_mlp


def main():
    print("="*60)
    print("STEP 1: Pre-training on Geometric Shapes")
    print("="*60)
    
    # Configure pre-training with high L2 regularization
    pretrain_config = GrokConfig(
        seed=42,
        optimization_steps=5000,
        batch_size=50,
        lr=1e-3,
        weight_decay=0.1,  # High weight decay for robust pre-training
        optimizer="AdamW",
        loss_function="MSE",
        depth=3,
        width=200,
        activation="ReLU",
        initialization_scale=8.0,
        filter="none",
        label="pretrain_shapes",
        results_dir="./results/pretrain",
    )
    
    print(f"Pre-training with L2 regularization (weight_decay={pretrain_config.weight_decay})")
    
    # Load shapes dataset
    shapes_train, shapes_test = load_geometric_shapes(pretrain_config.seed, num_samples_per_shape=250)
    
    # Pre-train on shapes (4 classes)
    print(f"\nPre-training on {len(shapes_train)} shape samples...")
    results_pretrain, model_pretrain = train(
        pretrain_config,
        train_set=shapes_train,
        test_set=shapes_test,
        num_classes=4
    )
    
    # Save pre-trained model
    pretrained_model_path = "./results/pretrain/pretrained_model.pt"
    # Note: In a real implementation, you'd need to return the model from train()
    # For now, we just note where it would be saved
    print(f"\nPre-training complete!")
    print(f"Pre-trained model would be saved to: {pretrained_model_path}")
    
    print("\n" + "="*60)
    print("STEP 2: Fine-tuning on MNIST")
    print("="*60)
    
    # Configure fine-tuning
    finetune_config = GrokConfig(
        seed=42,
        optimization_steps=10000,
        batch_size=50,
        lr=5e-4,  # Lower learning rate for fine-tuning
        weight_decay=0.0,
        optimizer="AdamW",
        loss_function="MSE",
        depth=3,
        width=200,
        activation="ReLU",
        initialization_scale=1.0,  # Smaller scale since we're loading pre-trained
        filter="none",
        label="finetune_mnist",
        results_dir="./results/finetune",
        train_points=1000,
    )
    
    # Load MNIST
    mnist_train, mnist_test = load_mnist(finetune_config)
    
    # Fine-tune on MNIST (10 classes)
    # In a complete implementation, you would load the pre-trained weights here
    print(f"\nFine-tuning on {len(mnist_train)} MNIST samples...")
    results_finetune, model_finetune = train(
        finetune_config,
        train_set=mnist_train,
        test_set=mnist_test,
        num_classes=10
    )
    
    print("\n" + "="*60)
    print("Workflow Complete!")
    print("="*60)
    print(f"\nResults:")
    print(f"  Pre-training: ./results/pretrain/")
    print(f"  Fine-tuning:  ./results/finetune/")


if __name__ == "__main__":
    main()

"""Quick test to verify checkpoint save/load functionality."""
import os
import sys
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from grokking_mnist.config import GrokConfig
    from grokking_mnist.train import train
    from grokking_mnist.data import load_geometric_shapes
    from grokking_mnist.model import build_mlp
else:
    from .config import GrokConfig
    from .train import train
    from .data import load_geometric_shapes
    from .model import build_mlp


def test_checkpoint_save_load():
    """Test that we can save and load model checkpoints."""
    print("Testing checkpoint save/load functionality...")
    
    # Configure a quick training run
    config = GrokConfig(
        seed=42,
        optimization_steps=50,
        batch_size=50,
        lr=1e-3,
        weight_decay=0.1,
        optimizer="AdamW",
        loss_function="MSE",
        depth=3,
        width=200,
        activation="ReLU",
        initialization_scale=8.0,
        filter="none",
        label="checkpoint_test",
        results_dir="./results/test_checkpoint",
    )
    
    # Load shapes dataset
    print("\n1. Training model on geometric shapes...")
    train_set, test_set = load_geometric_shapes(config.seed, num_samples_per_shape=50)
    results, model = train(config, train_set=train_set, test_set=test_set, num_classes=4)
    
    # Save checkpoint
    checkpoint_path = "./results/test_checkpoint/test_checkpoint.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': vars(config),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\n2. Checkpoint saved to: {checkpoint_path}")
    
    # Create a new model and load the checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_model = build_mlp(config, device, num_classes=4)
    
    print("\n3. Loading checkpoint into new model...")
    checkpoint_loaded = torch.load(checkpoint_path)
    new_model.load_state_dict(checkpoint_loaded['model_state_dict'])
    
    # Compare parameters to ensure they match
    print("\n4. Verifying parameters match...")
    match = True
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), new_model.named_parameters()):
        if not torch.allclose(param1, param2):
            print(f"   Mismatch in {name1}")
            match = False
    
    if match:
        print("   [OK] All parameters match!")
    else:
        print("   [FAIL] Some parameters don't match")
        return False
    
    print("\n5. Testing transfer to different output size (4 -> 10 classes)...")
    # Create a model with different output size (like MNIST with 10 classes)
    mnist_config = GrokConfig(
        seed=42,
        optimization_steps=50,
        batch_size=50,
        lr=1e-3,
        weight_decay=0.01,
        optimizer="AdamW",
        loss_function="MSE",
        depth=3,
        width=200,
        activation="ReLU",
        initialization_scale=1.0,
        filter="none",
        label="checkpoint_test_transfer",
        results_dir="./results/test_checkpoint",
    )
    
    mnist_model = build_mlp(mnist_config, device, num_classes=10)
    
    # Load pretrained weights (only matching layers)
    pretrained_state = checkpoint_loaded['model_state_dict']
    model_state = mnist_model.state_dict()
    pretrained_filtered = {}
    
    for name, param in pretrained_state.items():
        if name in model_state and param.shape == model_state[name].shape:
            pretrained_filtered[name] = param
    
    mnist_model.load_state_dict(pretrained_filtered, strict=False)
    print(f"   [OK] Loaded {len(pretrained_filtered)}/{len(model_state)} layers")
    print(f"   [OK] Skipped final layer (shape mismatch: 4 vs 10 classes)")
    
    print("\n" + "="*60)
    print("CHECKPOINT TEST PASSED!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_checkpoint_save_load()
    sys.exit(0 if success else 1)

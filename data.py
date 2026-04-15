import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

from .config import GrokConfig


def load_mnist(config: GrokConfig):
    """Load MNIST and return a small training subset + full test set.

    A small training subset is essential for inducing grokking: the model
    first memorises the few training samples, then (much later) generalises.
    """
    transform = transforms.ToTensor()

    train_full = torchvision.datasets.MNIST(
        root=config.download_directory,
        train=True,
        transform=transform,
        download=True,
    )
    test = torchvision.datasets.MNIST(
        root=config.download_directory,
        train=False,
        transform=transform,
        download=True,
    )

    train_subset = torch.utils.data.Subset(
        train_full, range(config.train_points)
    )
    return train_subset, test


def generate_line(size=28, thickness=2, angle=0, center_x=None, center_y=None):
    """Generate a line with specified angle and position."""
    img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(img)
    
    if center_x is None:
        center_x = size // 2
    if center_y is None:
        center_y = size // 2
    
    length = int(size * 0.6)
    
    angle_rad = np.radians(angle)
    x1 = center_x - length // 2 * np.cos(angle_rad)
    y1 = center_y - length // 2 * np.sin(angle_rad)
    x2 = center_x + length // 2 * np.cos(angle_rad)
    y2 = center_y + length // 2 * np.sin(angle_rad)
    
    draw.line([(x1, y1), (x2, y2)], fill=255, width=thickness)
    return np.array(img, dtype=np.float32) / 255.0


def generate_circle(size=28, radius=8, center_x=None, center_y=None):
    """Generate a circle with specified radius and position."""
    img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(img)
    
    if center_x is None:
        center_x = size // 2
    if center_y is None:
        center_y = size // 2
    
    bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
    draw.ellipse(bbox, fill=255)
    return np.array(img, dtype=np.float32) / 255.0


def generate_triangle(size=28, scale=1.0, angle=0, center_x=None, center_y=None):
    """Generate a triangle with specified scale, rotation, and position."""
    img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(img)
    
    if center_x is None:
        center_x = size // 2
    if center_y is None:
        center_y = size // 2
    
    base_size = int(size * 0.5 * scale)
    
    points = [
        (center_x, center_y - base_size),
        (center_x - base_size, center_y + base_size // 2),
        (center_x + base_size, center_y + base_size // 2),
    ]
    
    if angle != 0:
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotated_points = []
        for x, y in points:
            dx, dy = x - center_x, y - center_y
            new_x = center_x + dx * cos_a - dy * sin_a
            new_y = center_y + dx * sin_a + dy * cos_a
            rotated_points.append((new_x, new_y))
        points = rotated_points
    
    draw.polygon(points, fill=255)
    return np.array(img, dtype=np.float32) / 255.0


def generate_square(size=28, side_length=12, angle=0, center_x=None, center_y=None):
    """Generate a square with specified side length, rotation, and position."""
    img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(img)
    
    if center_x is None:
        center_x = size // 2
    if center_y is None:
        center_y = size // 2
    
    half = side_length // 2
    
    points = [
        (center_x - half, center_y - half),
        (center_x + half, center_y - half),
        (center_x + half, center_y + half),
        (center_x - half, center_y + half),
    ]
    
    if angle != 0:
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotated_points = []
        for x, y in points:
            dx, dy = x - center_x, y - center_y
            new_x = center_x + dx * cos_a - dy * sin_a
            new_y = center_y + dx * sin_a + dy * cos_a
            rotated_points.append((new_x, new_y))
        points = rotated_points
    
    draw.polygon(points, fill=255)
    return np.array(img, dtype=np.float32) / 255.0


def apply_sharpness(img, sigma):
    """Apply gaussian blur to control sharpness (higher sigma = less sharp)."""
    if sigma > 0:
        return ndimage.gaussian_filter(img, sigma=sigma)
    return img


class GeometricShapesDataset(torch.utils.data.Dataset):
    """Dataset of geometric shapes with variations in size, orientation, position, and sharpness.
    
    Shapes: 0=line, 1=circle, 2=triangle, 3=square
    
    Each shape appears at random locations within the 28x28 image with appropriate
    margins to prevent edge clipping.
    """
    
    def __init__(self, num_samples_per_shape=250, size=28, seed=42):
        self.size = size
        self.data = []
        self.labels = []
        
        np.random.seed(seed)
        
        shape_generators = [
            self._generate_line_variants,
            self._generate_circle_variants,
            self._generate_triangle_variants,
            self._generate_square_variants,
        ]
        
        for label, generator in enumerate(shape_generators):
            shapes = generator(num_samples_per_shape)
            self.data.extend(shapes)
            self.labels.extend([label] * len(shapes))
        
        self.data = torch.FloatTensor(np.array(self.data)).unsqueeze(1)
        self.labels = torch.LongTensor(self.labels)
    
    def _generate_line_variants(self, n):
        shapes = []
        for _ in range(n):
            thickness = np.random.choice([1, 2, 3])
            angle = np.random.uniform(0, 180)
            sharpness = np.random.choice([0, 0.3, 0.6, 1.0])
            
            # Random position with margins to avoid edge clipping
            margin = min(int(self.size * 0.35), self.size // 2 - 1)
            max_pos = max(margin + 1, self.size - margin)
            center_x = np.random.randint(margin, max_pos)
            center_y = np.random.randint(margin, max_pos)
            
            img = generate_line(self.size, thickness, angle, center_x, center_y)
            img = apply_sharpness(img, sharpness)
            shapes.append(img)
        return shapes
    
    def _generate_circle_variants(self, n):
        shapes = []
        for _ in range(n):
            radius = np.random.randint(4, 10)
            sharpness = np.random.choice([0, 0.3, 0.6, 1.0])
            
            # Random position with margins based on radius
            margin = min(radius + 2, self.size // 2 - 1)
            max_pos = max(margin + 1, self.size - margin)
            center_x = np.random.randint(margin, max_pos)
            center_y = np.random.randint(margin, max_pos)
            
            img = generate_circle(self.size, radius, center_x, center_y)
            img = apply_sharpness(img, sharpness)
            shapes.append(img)
        return shapes
    
    def _generate_triangle_variants(self, n):
        shapes = []
        for _ in range(n):
            scale = np.random.uniform(0.6, 1.2)
            angle = np.random.uniform(0, 360)
            sharpness = np.random.choice([0, 0.3, 0.6, 1.0])
            
            # Random position with margins based on scale
            base_size = int(self.size * 0.5 * scale)
            margin = min(base_size + 2, self.size // 2 - 1)
            max_pos = max(margin + 1, self.size - margin)
            center_x = np.random.randint(margin, max_pos)
            center_y = np.random.randint(margin, max_pos)
            
            img = generate_triangle(self.size, scale, angle, center_x, center_y)
            img = apply_sharpness(img, sharpness)
            shapes.append(img)
        return shapes
    
    def _generate_square_variants(self, n):
        shapes = []
        for _ in range(n):
            side_length = np.random.randint(8, 16)
            angle = np.random.uniform(0, 90)
            sharpness = np.random.choice([0, 0.3, 0.6, 1.0])
            
            # Random position with margins based on side length
            # Account for diagonal when rotated (multiply by sqrt(2))
            margin = min(int(side_length * 0.71) + 2, self.size // 2 - 1)
            max_pos = max(margin + 1, self.size - margin)
            center_x = np.random.randint(margin, max_pos)
            center_y = np.random.randint(margin, max_pos)
            
            img = generate_square(self.size, side_length, angle, center_x, center_y)
            img = apply_sharpness(img, sharpness)
            shapes.append(img)
        return shapes
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_geometric_shapes(config: GrokConfig, num_samples_per_shape=250):
    """Load geometric shapes dataset for pre-training.
    
    Returns train and test sets with 4 classes:
    0: lines (various angles, thicknesses, positions, sharpness)
    1: circles (various sizes, positions, sharpness)
    2: triangles (various sizes, orientations, positions, sharpness)
    3: squares (various sizes, orientations, positions, sharpness)
    
    All shapes appear at random locations within the image with appropriate margins.
    """
    train_set = GeometricShapesDataset(
        num_samples_per_shape=num_samples_per_shape,
        size=28,
        seed=config.seed
    )
    
    test_set = GeometricShapesDataset(
        num_samples_per_shape=num_samples_per_shape // 4,
        size=28,
        seed=config.seed + 1000
    )
    
    return train_set, test_set

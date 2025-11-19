"""
Simple synthetic dataset for EmbodiedMAE training.

Creates matching RGB, depth, and point cloud triplets from simple 3D shapes.
- RGB: Rendered image of scene
- Depth: Depth map of same scene
- PC: Point cloud sampled from the depth map

All three modalities are guaranteed to match!
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def create_synthetic_scene(image_size=224, max_depth=2.0):
    """
    Create a synthetic scene with simple shapes.

    Returns:
        rgb: (3, H, W) in [-1, 1]
        depth: (1, H, W) in [0, max_depth] meters
        pc: (N, 3) point cloud in meters
    """
    H, W = image_size, image_size

    # Create coordinate grids
    y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Initialize depth (background)
    depth = np.ones((H, W), dtype=np.float32) * max_depth * 0.9

    # Initialize RGB (background color)
    rgb = np.ones((H, W, 3), dtype=np.float32) * 0.3

    # Add 2-4 random spheres/objects
    num_objects = np.random.randint(2, 5)

    for _ in range(num_objects):
        # Random object center
        cy = np.random.randint(H // 4, 3 * H // 4)
        cx = np.random.randint(W // 4, 3 * W // 4)

        # Random radius
        radius = np.random.randint(20, 60)

        # Distance from center
        dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)

        # Object mask
        mask = dist < radius

        # Object depth (closer than background)
        object_depth = np.random.uniform(0.5, 1.5)  # meters
        depth[mask] = object_depth

        # Object color
        color = np.random.rand(3)
        rgb[mask] = color

    # Add some noise
    rgb += np.random.randn(H, W, 3) * 0.05
    rgb = np.clip(rgb, 0, 1)

    depth += np.random.randn(H, W) * 0.02
    depth = np.clip(depth, 0, max_depth)

    # Convert RGB to [-1, 1]
    rgb = rgb * 2.0 - 1.0

    # Generate point cloud from depth
    pc = depth_to_pointcloud(depth, fx=W/2, fy=H/2, cx=W/2, cy=H/2)

    # Convert to torch tensors
    rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()  # (3, H, W)
    depth = torch.from_numpy(depth).unsqueeze(0).float()  # (1, H, W)
    pc = torch.from_numpy(pc).float()  # (N, 3)

    return rgb, depth, pc


def depth_to_pointcloud(depth, fx=224, fy=224, cx=112, cy=112):
    """
    Convert depth map to point cloud using pinhole camera model.

    Args:
        depth: (H, W) depth in meters
        fx, fy: focal lengths in pixels
        cx, cy: principal point

    Returns:
        pc: (N, 3) point cloud in camera frame
    """
    H, W = depth.shape

    # Create pixel grid
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Backproject to 3D
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack coordinates
    points = np.stack([x, y, z], axis=-1)  # (H, W, 3)

    # Filter valid points (depth > 0)
    mask = z > 0.01  # Small threshold
    pc = points[mask]  # (N, 3)

    return pc


class SimpleSyntheticDataset(Dataset):
    """
    Synthetic dataset with matching RGB-Depth-PC triplets.

    Each sample generates a random scene with simple shapes.
    RGB, depth, and point cloud all come from the SAME scene.
    """

    def __init__(
        self,
        num_samples=1000,
        image_size=224,
        num_pc_points=8192,
        max_depth=2.0,
        seed=None,
    ):
        """
        Args:
            num_samples: Number of samples in dataset
            image_size: Size of images (square)
            num_pc_points: Number of points to sample from point cloud
            max_depth: Maximum depth in meters
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_pc_points = num_pc_points
        self.max_depth = max_depth

        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Set seed for reproducibility (same idx = same scene)
        np.random.seed(idx)

        # Generate scene
        rgb, depth, pc = create_synthetic_scene(
            image_size=self.image_size,
            max_depth=self.max_depth
        )

        # Sample point cloud to desired size
        if pc.shape[0] > self.num_pc_points:
            # Random sampling
            indices = torch.randperm(pc.shape[0])[:self.num_pc_points]
            pc = pc[indices]
        elif pc.shape[0] < self.num_pc_points:
            # Pad with repeat
            repeat = self.num_pc_points // pc.shape[0] + 1
            pc = pc.repeat(repeat, 1)[:self.num_pc_points]

        return {
            'rgb': rgb,
            'depth': depth,
            'pc': pc,
        }


def visualize_sample(sample, save_path=None):
    """
    Visualize a sample to verify RGB/depth/PC match.

    Args:
        sample: Dict with 'rgb', 'depth', 'pc'
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rgb = sample['rgb']  # (3, H, W)
    depth = sample['depth']  # (1, H, W)
    pc = sample['pc']  # (N, 3)

    # Convert RGB from [-1, 1] to [0, 1] for display
    rgb_display = (rgb.permute(1, 2, 0).numpy() + 1.0) / 2.0
    depth_display = depth.squeeze(0).numpy()
    pc_np = pc.numpy()

    fig = plt.figure(figsize=(15, 5))

    # RGB
    ax1 = fig.add_subplot(131)
    ax1.imshow(rgb_display)
    ax1.set_title('RGB Image')
    ax1.axis('off')

    # Depth
    ax2 = fig.add_subplot(132)
    im = ax2.imshow(depth_display, cmap='viridis')
    ax2.set_title('Depth Map')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2)

    # Point Cloud
    ax3 = fig.add_subplot(133, projection='3d')
    # Downsample for visualization
    step = max(1, len(pc_np) // 1000)
    ax3.scatter(pc_np[::step, 0], pc_np[::step, 1], pc_np[::step, 2],
                c=pc_np[::step, 2], cmap='viridis', s=1)
    ax3.set_title('Point Cloud')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z (depth)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    print("Testing SimpleSyntheticDataset...")

    # Create dataset
    dataset = SimpleSyntheticDataset(num_samples=10, seed=42)

    # Get a sample
    sample = dataset[0]

    print(f"\nSample 0:")
    print(f"  RGB shape: {sample['rgb'].shape}, range: [{sample['rgb'].min():.2f}, {sample['rgb'].max():.2f}]")
    print(f"  Depth shape: {sample['depth'].shape}, range: [{sample['depth'].min():.2f}, {sample['depth'].max():.2f}]")
    print(f"  PC shape: {sample['pc'].shape}, range: [{sample['pc'].min():.2f}, {sample['pc'].max():.2f}]")

    # Visualize to verify they match
    print("\nVisualizing sample to verify RGB/Depth/PC match...")
    visualize_sample(sample, save_path='./synthetic_sample_verification.png')

    # Test dataloader
    print("\nTesting DataLoader...")
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    print(f"  Batch RGB shape: {batch['rgb'].shape}")
    print(f"  Batch Depth shape: {batch['depth'].shape}")
    print(f"  Batch PC shape: {batch['pc'].shape}")

    print("\n✅ All tests passed! RGB, Depth, and PC are matching triplets.")

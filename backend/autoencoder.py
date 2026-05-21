"""
Convolutional Autoencoder for Image Compression.

Architecture:
    Encoder: 3 Conv2d layers → compressed latent representation
    Decoder: 3 ConvTranspose2d layers → reconstructed image

Why this works as compression:
    - Input:   3 × 256 × 256 = 196,608 values
    - Latent:  64 × 32 × 32  =  65,536 values  → 3x compression
    - The encoder learns WHICH features matter most (like DCT but learned)
    - The decoder learns HOW to reconstruct from those features

Difference from JPEG:
    - JPEG uses fixed DCT basis functions (mathematical, not learned)
    - This autoencoder LEARNS its own basis functions from training data
    - That's what makes it "AI compression" — it adapts to image content
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Compresses image to latent representation.

    Input:  (B, 3, H, W)        — RGB image, any size (will be resized externally)
    Output: (B, 64, H/8, W/8)   — compressed latent
    """
    def __init__(self, latent_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: 3 → 32 channels, downsample ÷2
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 2: 32 → 64 channels, downsample ÷2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 3: 64 → latent_channels, downsample ÷2
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """
    Reconstructs image from latent representation.

    Input:  (B, 64, H/8, W/8)  — compressed latent
    Output: (B, 3, H, W)       — reconstructed RGB image (values 0-1)
    """
    def __init__(self, latent_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: latent_channels → 64, upsample ×2
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 2: 64 → 32, upsample ×2
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 3: 32 → 3, upsample ×2 — output is full image
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid(),   # clamp output to [0, 1]
        )

    def forward(self, x):
        return self.net(x)


class ImageAutoencoder(nn.Module):
    """
    Full autoencoder = Encoder + Decoder.
    Used during training only.
    At inference, Encoder and Decoder are used separately.
    """
    def __init__(self, latent_channels: int = 64):
        super().__init__()
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)
        self.latent_channels = latent_channels

    def forward(self, x):
        latent = self.encoder(x)
        recon  = self.decoder(latent)
        return recon, latent

    def compress(self, x):
        """Return latent tensor — this is the compressed representation."""
        with torch.no_grad():
            return self.encoder(x)

    def decompress(self, latent):
        """Reconstruct image from latent tensor."""
        with torch.no_grad():
            return self.decoder(latent)

    def compression_ratio(self, input_size: int = 256) -> float:
        """
        Calculate theoretical compression ratio.
        input_size: square image side length in pixels
        """
        input_vals  = 3 * input_size * input_size
        latent_vals = self.latent_channels * (input_size // 8) * (input_size // 8)
        return input_vals / latent_vals


class CombinedLoss(nn.Module):
    """
    MSE loss + perceptual loss component.

    MSE alone tends to produce blurry reconstructions.
    Adding a gradient-based sharpness term helps preserve edges.

    total_loss = mse_weight * MSE + grad_weight * GradientLoss
    """
    def __init__(self, mse_weight: float = 1.0, grad_weight: float = 0.1):
        super().__init__()
        self.mse       = nn.MSELoss()
        self.mse_w     = mse_weight
        self.grad_w    = grad_weight

    def gradient_loss(self, pred, target):
        """Penalize difference in image gradients — preserves edges."""
        def grad(x):
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]   # horizontal
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]   # vertical
            return dx, dy

        pred_dx,   pred_dy   = grad(pred)
        target_dx, target_dy = grad(target)

        return (self.mse(pred_dx, target_dx) +
                self.mse(pred_dy, target_dy)) / 2

    def forward(self, pred, target):
        mse_loss  = self.mse(pred, target)
        grad_loss = self.gradient_loss(pred, target)
        return self.mse_w * mse_loss + self.grad_w * grad_loss


if __name__ == '__main__':
    # Quick sanity check
    model = ImageAutoencoder(latent_channels=64)
    dummy = torch.randn(2, 3, 256, 256)   # batch of 2 images

    recon, latent = model(dummy)

    print("=" * 50)
    print("  Autoencoder Architecture Check")
    print("=" * 50)
    print(f"  Input shape  : {list(dummy.shape)}")
    print(f"  Latent shape : {list(latent.shape)}")
    print(f"  Output shape : {list(recon.shape)}")
    print(f"  Compression  : {model.compression_ratio(256):.2f}x")
    print()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params : {total_params:,}")
    print("=" * 50)
    print("  ✅ Architecture OK!")
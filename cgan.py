import torch.nn.functional as F
from layers import DownBlock, UpBlock, GANUpBlock
import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, latent_dim=64, num_filters=64, im_dim=None, num_classes=None, num_channels=1, activation=nn.ReLU(inplace=True)):
        """
        classify image batch: BxCx{im_dim}x{im_dim}
        outputs: Bx1
        """
        super().__init__()

        self.blocks = nn.Sequential(
            DownBlock(num_channels * 2 , num_filters, first_layer=True, activation=activation),                                  # 28 -> 16
            DownBlock(num_filters, num_filters * 2, stride=2, activation=activation),                                            # 16 -> 8
            DownBlock(num_filters * 2, num_filters * 4, stride=2, activation=activation),                                        # 8 -> 4
            DownBlock(num_filters * 4, num_filters * 4, stride=2, activation=activation),                                        # 4 -> 2
            DownBlock(num_filters * 4, 1, stride=2, activation=activation)                                                       # 2 -> 1
        )

        self.class_embeds = nn.Embedding(num_classes, int(im_dim ** 2))
        
        self.latent_dim = latent_dim
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.im_dim = im_dim
        
    def forward(self, x, targets, device='cpu'):
        
        B = x.shape[0]

        x = torch.cat(
            [x, self.class_embeds(targets).reshape(B, 1, self.im_dim, self.im_dim)],
            dim=1
        )

        logits = self.blocks(x).reshape(B, 1)

        return logits
    
class Generator(nn.Module):
    def __init__(self, latent_dim=64, num_filters=64, im_dim=None, num_classes=None, num_channels=1, activation=nn.ReLU(inplace=True)):
        """
        recontructs noise batch: Bx{latent_dim}x1x1
        into image BxCx{im_dim}x{im_dim}
        """
        super().__init__()

        # Bx{latent_dim}x1x1
        self.blocks = nn.Sequential(
            GANUpBlock(
                upsample_layer=nn.ConvTranspose2d(2 * latent_dim, num_filters * 16, 4, 2, 1),
                first_layer=True,
                activation=activation
            ),                                                                                                      # Bx{16*num_filters}x2x2
            GANUpBlock(
                upsample_layer=nn.ConvTranspose2d(num_filters * 16, num_filters * 8, 4, 2, 1),
                activation=activation
            ),                                                                                                      # Bx{8xnum_filters}x4x4
            GANUpBlock(
                upsample_layer=nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 4, 2, 1),
                activation=activation
            ),                                                                                                      # Bx{4xnum_filters}x8x8                      
            GANUpBlock(
                upsample_layer=nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 4, 2, 1),
                activation=activation
            ),                                                                                                      # Bx{2xnum_filters}x16x16
            GANUpBlock(
                upsample_layer=nn.ConvTranspose2d(num_filters * 2, num_channels, 2, 2, 2),
                last_layer=True,
                activation=activation
            ),                                                                                                      # Bx{num_filters}x28x28   
            nn.Tanh()
            # Bx{C}x{im_dim}x{im_dim}     
        )

        self.class_embeds = nn.Embedding(num_classes, latent_dim)
        
        self.latent_dim = latent_dim
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.im_dim = im_dim
        
    def forward(self, batch_size, targets, device='cpu'):
        
        x = torch.randn(batch_size, self.latent_dim).to(device)

        z = torch.cat([self.class_embeds(targets), x], dim=1)
    
        return self.blocks(z.reshape(*z.shape, 1, 1))
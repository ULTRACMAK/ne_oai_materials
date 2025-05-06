from layers import DownBlock, UpBlock, HiddenBlock, StatisticsBlock

import torch.nn.functional as F
import torch.nn as nn
import torch


class CVAE(nn.Module):
    def __init__(self, num_classes=None, hidden_dim=None, num_filters=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_filters = num_filters

        self.encoder = nn.Sequential(
            DownBlock(1, self.num_filters, first_layer=True),                                                       # 28 -> 16
            DownBlock(self.num_filters, self.num_filters * 2, stride=2),                                            # 16 -> 8
            DownBlock(self.num_filters * 2, self.num_filters * 4, stride=2),                                        # 8 -> 4
            DownBlock(self.num_filters * 4, self.num_filters * 8, stride=2),                                        # 4 -> 2
            DownBlock(self.num_filters * 8, self.num_filters * 16, stride=2)                                        # 2 -> 1
        )

        self.hidden_mapping = StatisticsBlock(self.num_filters * 16, self.hidden_dim)

        self.decoder = nn.Sequential(
            UpBlock(
                self.hidden_dim + self.hidden_dim, self.num_filters * 16, first_layer=True,
                upsample_layer=nn.Upsample(scale_factor=2, mode='bilinear')
            ),                                                                                                      # 1 -> 2
            UpBlock(
                self.num_filters * 16, self.num_filters * 8,
                upsample_layer=nn.Upsample(scale_factor=2, mode='bilinear')
            ),                                                                                                      # 2 -> 4
            UpBlock(
                self.num_filters * 8, self.num_filters * 4,
                upsample_layer=nn.Upsample(scale_factor=2, mode='bilinear')
            ),                                                                                                      # 4 -> 8                      
            UpBlock(
                self.num_filters * 4, self.num_filters * 2,
                upsample_layer=nn.Upsample(scale_factor=2, mode='bilinear')
            ),                                                                                                      # 8 -> 16
            UpBlock(
                self.num_filters * 2, 1, last_layer=True,
                upsample_layer=nn.Upsample(size=28, mode='bilinear')
            )                                                                                                       # 16 -> 28        
        )

        self.class_embed = nn.Embedding(num_classes, hidden_dim)

    def get_latent(self, x):
        hidden_state = self.encoder(x).squeeze()

        mu, log_sigma = self.hidden_mapping(hidden_state)
        
        z = self.reparametrize(mu, log_sigma)
        
        return z
        
    def forward(self, x, targets):       
        
        hidden_state = self.encoder(x).squeeze()
        
        mu, log_sigma = self.hidden_mapping(hidden_state)

        z = self.reparametrize(mu, log_sigma)
        
        # ohe = F.one_hot(targets, num_classes=self.num_classes)
        target_embed = self.class_embed(targets)
        
        z = torch.cat([z, target_embed], 1)
        z = z.reshape(*z.shape, 1, 1)

        x = self.decoder(z)
        reconstruction = torch.tanh(x)

        return mu, log_sigma, reconstruction
    
    def sample(self, hidden_state, targets, device=None):
        # ohe = F.one_hot(targets, num_classes=self.num_classes).to(device)
        target_embed = self.class_embed(targets)
        
        z = torch.cat([hidden_state.to(device), target_embed], 1)
        z = z.reshape(*z.shape, 1, 1)

        x = self.decoder(z)
        return torch.tanh(x)
    
    def reparametrize(self, mu, log_sigma):
        # ln( sigma^2 ) = x
        # 2 ln( sigma ) = x
        # sigma = e ^ (x / 2) 
        
        z = mu
        
        if self.training:
            sigma = (log_sigma / 2).exp()
            z += torch.randn_like(sigma) * sigma
        
        return z
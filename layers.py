import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(
        self, in_channels=None, out_channels=None, padding=1, stride=1,
        last_layer=False, first_layer=False, activation=nn.ReLU(inplace=True)
        ):
        """
        Downsampling block
        """
        super().__init__()
        
        # stride == N allows us for compress image by N
        # compensate with pad for kernel size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4 if not last_layer else 1,
            padding=padding if not last_layer else 0,
            bias=first_layer or last_layer, # as far as we switch off BN and activation,
            stride=stride
        ) 

        self.act = activation
        self.dropout = nn.Dropout(0.1)

        # cancel bias term in case that we have it in convolution
        self.norm = nn.BatchNorm2d(
            num_features=out_channels
        ) if not (first_layer or last_layer) else None
        
        self.last_layer = last_layer
        self.first_layer = first_layer
        
    def forward(self, x):
        
        x = self.conv(x)
        
        if self.first_layer:
            x = self.act(x)
        elif not self.last_layer:
            x =  self.dropout(self.act(self.norm(x)))
    
        return x

class HiddenBlock(nn.Module):
    def __init__(self, out_dim=None, hidden_dim=None):
        super().__init__()

        self.linear = nn.Linear(out_dim, hidden_dim)

    def forward(self, x):

        return self.linear(x)

class StatisticsBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        
        # Here we output 2*latent_space because we want to receive 2 vectors of mu and sigma
        # as you might guessed output logits could be negative
        # so we should treat them as logarithms in case of sigma(which should be positive)
        self.fc_mu = nn.Linear(c_in, c_out)
        self.fc_sigma = nn.Linear(c_in, c_out)

    def forward(self, x):
        
        mu = self.fc_mu(x)
        log_sigma = self.fc_sigma(x)

        return mu, log_sigma

class UpBlock(nn.Module):
    def __init__(
        self, in_channels=None, out_channels=None, upsample_layer=None,
        last_layer=False, first_layer=False, activation=nn.ReLU(inplace=True)
        ):
        """
        Upsampling block
        """
        super().__init__()

        self.upsample = upsample_layer
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

        self.act = activation
        self.dropout = nn.Dropout(0.2)

        # cancel bias term in case that we have it in convolution
        self.norm = nn.BatchNorm2d(
            num_features=out_channels
        ) if not (first_layer or last_layer) else None
        
        self.last_layer = last_layer
        self.first_layer = first_layer
        
    def forward(self, x):

        x = self.upsample(x)
        
        x = self.conv(x)
        
        if self.first_layer:
            x = self.act(x)
        elif not self.last_layer:
            x = self.dropout(self.act(self.norm(x)))
    
        return x
    
class GANUpBlock(nn.Module):
    def __init__(
        self, upsample_layer=None,
        last_layer=False, first_layer=False,
        activation=nn.ReLU(inplace=True)
        ):
        """
        Upsampling block
        """
        super().__init__()

        # self.upsample = upsample_layer
        
        self.conv = upsample_layer

        self.act = activation
        self.dropout = nn.Dropout(0.2)

        # cancel bias term in case that we have it in convolution
        self.norm = nn.BatchNorm2d(
            num_features=self.conv.out_channels
        ) if not (first_layer or last_layer) else None
        
        self.last_layer = last_layer
        self.first_layer = first_layer
        
    def forward(self, x):
        
        x = self.conv(x)
        
        if self.first_layer:
            x = self.act(x)
        elif not self.last_layer:
            x = self.dropout(self.act(self.norm(x)))
    
        return x
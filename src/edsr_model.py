import torch
import torch.nn as nn
import math

class ResidualBlock(nn.Module):
    """Enhanced Residual Block for EDSR"""
    
    def __init__(self, channels=32, res_scale=0.1):
        super(ResidualBlock, self).__init__()
        
        self.res_scale = res_scale
        
        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = out * self.res_scale
        out = torch.add(out, residual)
        
        return out

class EDSR(nn.Module):
    """
    Enhanced Deep Super-Resolution Network (EDSR)
    Modified for same input/output size
    """
    
    def __init__(self, num_channels=3, num_features=32, num_blocks=4, res_scale=0.1):
        super(EDSR, self).__init__()
        
        # Initial feature extraction
        self.conv_input = nn.Conv2d(
            num_channels, num_features,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(num_features, res_scale)
            for _ in range(num_blocks)
        ])
        
        # Global skip connection
        self.conv_mid = nn.Conv2d(
            num_features, num_features,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        # Reconstruction layer - output same size as input
        self.conv_output = nn.Conv2d(
            num_features, num_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        """Forward pass - maintains same input/output size"""
        # Initial feature extraction
        out = self.conv_input(x)
        
        # Store for skip connection
        skip = out
        
        # Residual blocks
        out = self.residual_blocks(out)
        
        # Global skip connection
        out = self.conv_mid(out)
        out = torch.add(out, skip)
        
        # Reconstruction
        out = self.conv_output(out)
        
        return out
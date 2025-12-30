import torch
import torch.nn as nn

class SRCNN(nn.Module):
    """
    Super-Resolution Convolutional Neural Network (SRCNN)
    Original paper: "Image Super-Resolution Using Deep Convolutional Networks"
    """
    
    def __init__(self, num_channels=3, base_filter=64):
        super(SRCNN, self).__init__()
        
        # Feature extraction layer
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=base_filter,
            kernel_size=9,
            padding=4,
            bias=True
        )
        
        # Non-linear mapping layer
        self.conv2 = nn.Conv2d(
            in_channels=base_filter,
            out_channels=base_filter//2,
            kernel_size=5,
            padding=2,
            bias=True
        )
        
        # Reconstruction layer
        self.conv3 = nn.Conv2d(
            in_channels=base_filter//2,
            out_channels=num_channels,
            kernel_size=5,
            padding=2,
            bias=True
        )
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # Feature extraction
        x = self.relu(self.conv1(x))
        
        # Non-linear mapping
        x = self.relu(self.conv2(x))
        
        # Reconstruction
        x = self.conv3(x)
        
        return x
    
    def get_complexity(self):
        """Calculate model complexity"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'layers': 3
        }
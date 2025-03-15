import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

class Hourglass(nn.Module):
    def __init__(self, depth, num_features):
        super().__init__()
        self.depth = depth
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        
        for _ in range(depth):
            self.down_layers.append(ResidualBlock(num_features, num_features))
        
        for _ in range(depth):
            self.up_layers.append(ResidualBlock(num_features, num_features))

    def forward(self, x):
        down_outputs = []
        for layer in self.down_layers:
            x = layer(x)
            down_outputs.append(x)
            x = F.max_pool2d(x, 2)
        
        for i in range(self.depth):
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.up_layers[i](x + down_outputs[self.depth - 1 - i])
        
        return x

class StackedHourglassNetwork(nn.Module):
    def __init__(self, num_stacks=3, num_features=256, num_output_points=4):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            ResidualBlock(128, num_features)
        )
        
        self.stacks = nn.ModuleList([Hourglass(4, num_features) for _ in range(num_stacks)])
        self.intermediate_convs = nn.ModuleList([nn.Conv2d(num_features, num_features, kernel_size=1) for _ in range(num_stacks)])
        self.output_layers = nn.ModuleList([nn.Conv2d(num_features, num_output_points, kernel_size=1) for _ in range(num_stacks)])
    
    def forward(self, x):
        x = self.preprocess(x)
        outputs = []
        for i in range(len(self.stacks)):
            x = self.stacks[i](x)
            outputs.append(self.output_layers[i](x))
            if i < len(self.stacks) - 1:
                x = self.intermediate_convs[i](x)

        # Put outputs in tensor
        outputs = torch.stack(outputs, dim=1)
        
        return outputs  # Output shape: (N, num_stacks, num_output_points, H, W)

# Example usage
if __name__ == '__main__':
    model = StackedHourglassNetwork(num_stacks=3, num_features=256, num_output_points=1)
    input_tensor = torch.randn(8, 3, 256, 256) # Example batch of 8 images
    outputs = model(input_tensor)
    print(outputs.shape) # Expected: (8, 3, 1, H, W) representing 8 batch outputs of 3 hourglass outputs of 1 heatmap
    for output in outputs:
        print(output.shape) # Expected: (3, 1, H, W) representing 3 hourglass outputs of 1 heatmaps
        print(output[0].shape) # Expected: (1, H, W) representing 1 heatmap
        print(output[0, 0].shape) # Expected: (H, W) representing that heatmap
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

class DilatedResBlock(nn.Module):
    def __init__(self, main_channels, kernel, padding, dilation=1, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(main_channels, main_channels, kernel, padding=padding, dilation=dilation, bias=False)
        self.gn1 = nn.GroupNorm(groups, main_channels)
        self.conv2 = nn.Conv2d(main_channels, main_channels, kernel, padding=padding, dilation=dilation, bias=False)
        self.gn2 = nn.GroupNorm(groups, main_channels)
    
    def forward(self, x):
        h = F.silu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.silu(x + h)

class DilatedResLayer(nn.Module):
    def __init__(self, main_channels, num_layers, kernel, padding, dilation, groups):
        super().__init__()
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([])
        for _ in range(num_layers):
            self.blocks.append(DilatedResBlock(main_channels=main_channels, kernel=kernel, padding=padding, dilation=dilation, groups=groups))    
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.blocks[i](x)
        return x

class MapEncoder(nn.Module):
    def __init__(self, in_channels, main_channels, out_channels, num_blocks: List, kernel_list: List, padding_list: List, dilation_list: List, groups: int, dropout: float):
        super().__init__()
        
        assert len(num_blocks) == 4 and len(dilation_list) == 4, "argument num_blocks, dilation_list must have length: 4"
            
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, main_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, main_channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            DilatedResLayer(main_channels, num_layers=num_blocks[0], kernel=kernel_list[0], padding=padding_list[0], dilation=dilation_list[0], groups=groups), # Stage 1
            nn.Dropout(p=dropout),
            DilatedResLayer(main_channels, num_layers=num_blocks[1], kernel=kernel_list[1], padding=padding_list[1], dilation=dilation_list[1], groups=groups), # Stage 2
            nn.Dropout(p=dropout),
            DilatedResLayer(main_channels, num_layers=num_blocks[2], kernel=kernel_list[2], padding=padding_list[2], dilation=dilation_list[2], groups=groups), # Stage 3
            nn.Dropout(p=dropout),
            DilatedResLayer(main_channels, num_layers=num_blocks[3], kernel=kernel_list[3], padding=padding_list[3], dilation=dilation_list[3], groups=groups), # Stage 4
        )
        self.out = nn.Conv2d(main_channels, out_channels, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        f = self.out(self.blocks(self.stem(x))) # (B, out_channels, 8, 8)
        g = self.pool(f).flatten(1) # (B, out_channels)
        return f, g
    

def main():
    in_channels = 3
    main_channels = 512
    out_channels = 256
    num_blocks = [2, 2, 6, 3]
    dilation_list = [1, 2, 4, 1]
    groups = 8
    
    ex_data = torch.randn((32, 3, 8, 8))
    map_encoder = MapEncoder(in_channels=in_channels, main_channels=main_channels,
                             out_channels=out_channels, num_blocks=num_blocks, dilation_list=dilation_list,
                             groups=groups)
    data_out = map_encoder(ex_data)
    
    print("data out shape: ", data_out[0].shape, data_out[1].shape)
    
    from torchinfo import summary
    summary(map_encoder)

if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
from typing import Union

from core.diffuser.networks.helpers import Downsample1d, Upsample1d, Conv1dBlock
from core.diffuser.networks.helpers import SinusoidalPosEmb
from core.diffuser.networks.vit import ViT, TransformerEncoder, TransformerDecoder
from core.diffuser.networks.MLP import MLP
from core.diffuser.networks.CNN import CNN

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

# This contains map encoder & decoder
# UNet architecture for 1D signals with conditional inputs
class ConditionalUnet1DTransP(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        network_config={},
        is_cnn=False,
    ):
        """
        input_dim: Dim of actions.
        cond_dim: Dim of conditionals.
        down_dims: Channel size for each UNet level.
          The length of this array determines number of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """
  
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )
        
        # latent transformer
        latent_trans_config = network_config['latent_transformer_config']
        self.latent_transformer = TransformerDecoder(
            dim=cond_dim,
            depth=latent_trans_config['depth'],
            heads=latent_trans_config['heads'],
            dim_head=cond_dim // latent_trans_config['heads'],
            mlp_dim=latent_trans_config['mlp_dim'],
            dropout=latent_trans_config['dropout'],
        )
        self.latent_scale = nn.Sequential(
            nn.Linear(mid_dim, mid_dim//8),
            nn.ReLU(),
            nn.Linear(mid_dim//8, mid_dim),
            nn.Sigmoid()
        )
        
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )
        
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
         
        vit_embed_dim = network_config['vit_config']['num_classes']
        obs_embed_dim = network_config['mlp_config']['embed_dim']
        cnn_embed_dim = network_config['cnn_config']['output_dim']
        if not is_cnn:
            if vit_embed_dim != obs_embed_dim:
                raise Exception("Dimension of map feature and observation feature must be same! (ViT and MLP)")
        else:
            if cnn_embed_dim != obs_embed_dim:
                raise Exception("Dimension of map feature and observation feature must be same! (CNN and MLP)")
            
        # 1. Condition: binary Map
        if is_cnn:
            self.map_encoder = CNN(input_dim=network_config['cnn_config']['input_dim'], output_dim=network_config['cnn_config']['output_dim'])
        else:
            self.vit_config = network_config['vit_config']
            self.map_encoder = ViT(**self.vit_config) # for ex, output 32 dim (num_classes in config)
        
        # 2. Condition: observation (start, goal location)
        self.mlp_config = network_config['mlp_config']
        self.env_cond = MLP(**self.mlp_config) # for ex, output 64 dim
        
        # 3. Condition: diffusion learnable time-step token
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(vit_embed_dim),
            nn.Linear(vit_embed_dim, vit_embed_dim * 4),
            nn.Mish(),
            nn.Linear(vit_embed_dim * 4, vit_embed_dim),
        )
        
        # 4. Condition Encoder: transformer-self attention that learns correlation between conditions
        condition_encoder_config = {
            "dim": vit_embed_dim,
            "depth": 2,
            "heads": 1,
            "dim_head": vit_embed_dim//1,
            "mlp_dim": vit_embed_dim * 4,
            "dropout": 0.1
        }
        self.condition_encoder = TransformerEncoder(**condition_encoder_config)
        
        # 5. adapter network that transform transformer to convolution based feature representation
        self.cond_adapter = nn.Sequential(
            Conv1dBlock(inp_channels=vit_embed_dim, out_channels=cond_dim, kernel_size=1, n_groups=1),
            Conv1dBlock(inp_channels=cond_dim, out_channels=cond_dim, kernel_size=kernel_size, n_groups=n_groups)
        )
        
        # 6. compress latent dimension to condition dimension
        self.latent_compressor = nn.Sequential(
            Conv1dBlock(inp_channels=mid_dim, out_channels=cond_dim, kernel_size=1, n_groups=1),
            Conv1dBlock(inp_channels=cond_dim, out_channels=cond_dim, kernel_size=kernel_size, n_groups=n_groups)
        )
        

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        map_cond=None,
        env_cond=None
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        map_cond: (B,map_cond(B, 1, 8, 8))
        env_cond: (B, env_cond(B, start: {obs_dim}, goal: {obs_dim}))
        output: (B,T,input_dim)
        """
        
        # 1. adjust sample tensor format for conv1d
        # (B,T,C)
        sample = sample.moveaxis(-1, -2)
        # (B,C,T)
        
        # 2. encode map & action feature and learn the correlation between them
        map_cond = self.map_encoder(map_cond) # (B, 16, 64) in case of 8x8 size, 2x2 patch, vit num_cls=64

        # 3. encode observation (start, end goal) state
        env_cond = self.env_cond(env_cond) # (B, 2*2) -> (B, 64(=mlp embed_dim))
        env_cond = env_cond.reshape(env_cond.shape[0], 1, env_cond.shape[1]) # (B, 1, 64)

        # 4, conditions: map and obs state
        conditionals = torch.cat([map_cond, env_cond], dim=1) # (B, 16+1, 64)

        # 5. encode learnable time-step.
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        
        # 6. add pos embedding (sinusoidal) and learn the feature
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_embed = self.diffusion_step_encoder(timesteps) # (B, diffusion_dim)
        time_embed = time_embed.unsqueeze(1) # (B, 1, diffusion_dim)

        # 7. add timestep embedding to conditionals.
        conditionals = torch.concat([conditionals, time_embed], dim=1) # (B, 16+1+1, 64)
        
        # 8. learn correlation between conditions.
        conditionals = self.condition_encoder(conditionals)
        
        # 9. adapt feature to convolution based feature representation
        conditionals = conditionals.permute(0, 2, 1) # (B, T, C) -> (B, C, T)
        conditionals = self.cond_adapter(conditionals)
        condition_1d = torch.mean(conditionals, dim=-1) # (B, C, T) -> (B, C)
        
        x = sample
        h = []
        for _, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, condition_1d)
            x = resnet2(x, condition_1d)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, condition_1d)
        
        # 10. update conditional information with latent transformer (cross-attention)
        compressed_latent = self.latent_compressor(x).permute(0, 2, 1) # (B, C, T) -> (B, T, C)
        conditionals = self.latent_transformer(conditionals.permute(0, 2, 1), compressed_latent)
        cond_1d_updated = torch.mean(conditionals.permute(0, 2, 1), dim=-1) # (B, T, C) -> (B, C, T) -> (B, C)
        
        for _, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, cond_1d_updated)
            x = resnet2(x, cond_1d_updated)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x

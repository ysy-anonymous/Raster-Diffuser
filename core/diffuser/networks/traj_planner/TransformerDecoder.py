import torch
import torch.nn as nn
from typing import Union

from core.diffuser.networks import CNN
from core.diffuser.networks.vit import TransformerDecoder, TransformerEncoder, ViT
from core.diffuser.networks.helpers import SinusoidalPosEmb
from core.diffuser.networks.MLP import MLP

class TransformerPathGenerator(nn.Module):
    def __init__(
        self,
        traj_dim: int, # input trajectory dimension (=same as action_dim = 2)
        network_config: dict = {},
        is_cnn: bool = False,
    ):
        """
        Diffusion Path Generator based on transformer decoder architecture.

        Args:
            input_dim: noisy trajectory dimension (action_dim)
            diffusion_dim: diffusion main path dimension
            num_heads: Number of attention heads in transformer decoder
            n_groups: Number of groups for GroupNorm
            network_config: Configuration dictionary for transformer decoder
        """
        super().__init__()
        
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
            self.map_encoder = CNN(input_dim=1, output_dim=network_config['cnn_config']['output_dim']) # for ex, output 32 dim (output_dim in config)
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
        
        # 5. Adjust Condition dimension to same as diffusion main dim
        transformer_decoder_conf = network_config['transformer_decoder_config']
        diffusion_dim = transformer_decoder_conf['dim_head'] * transformer_decoder_conf['heads']
        self.cond2diff = nn.Sequential(
            nn.Linear(vit_embed_dim, vit_embed_dim * 4),
            nn.Mish(),
            nn.Linear(vit_embed_dim * 4, vit_embed_dim * 4),
            nn.Mish(),
            nn.Linear(vit_embed_dim * 4, diffusion_dim)
        )
        
        # noisy-trajectory sample encoder (adjust dimension according to diffusion main dimension)
        self.sample_encoder = nn.Sequential(
            nn.Linear(traj_dim, 64),
            nn.Mish(),
            nn.Linear(64, 128),
            nn.Mish(),
            nn.Linear(128, diffusion_dim)
        )

        # transformer decoder -> cross-attention and ffn layers
        self.transformer_decoder = TransformerDecoder(
            dim=diffusion_dim,
            depth=transformer_decoder_conf['depth'],
            heads=transformer_decoder_conf['heads'],
            dim_head=transformer_decoder_conf['dim_head'],
            mlp_dim=transformer_decoder_conf['mlp_dim'],
            dropout=transformer_decoder_conf['dropout'],
        )

        # final layer that maps diffusion main dimension back to trajectory dimension (=action_dim=2)
        self.planning_head = nn.Sequential(
            nn.Linear(diffusion_dim, diffusion_dim * 2),
            nn.Mish(),
            nn.Linear(diffusion_dim * 2, diffusion_dim),
            nn.Mish(),
            nn.Linear(diffusion_dim, traj_dim)
        )
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        map_cond = None,
        env_cond = None,
    ):
        """
        x: (B, T, input_dim)-> (B, 32, 2) noisy trajectory
        timestep: (B, ) or int, diffusion step
        map_cond: (B, 1, 8, 8)
        env_cond: (B, 4)
        output: (B, T, input_dim)
        """

        # 1. encode noisy trajectory sampling: (B, T, input_dim) form, (B, 32, 2) -> (B, 32, diffusion_dim)
        sample = self.sample_encoder(sample)
        
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
        
        # 9. bridge between conditional feature and diffusion conditionals
        conditionals = self.cond2diff(conditionals)
        
        # 10. cross-attention between encoded sample and conditionals
        x = self.transformer_decoder(sample, conditionals)

        # 11. trajectory planning mlp-head
        x = self.planning_head(x) # (B, T, diffusion_dim) -> (B, T, traj_dim(=2))
        
        # print("check sample output: ", x[0, :, :])
        
        return x
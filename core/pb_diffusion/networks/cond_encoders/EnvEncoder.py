import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.normalizer import LinearNormalizer

# Observation Embedding layer using nn.Embedding
class OBSEmbedEncoder(nn.Module):
    """
        obs_dim: observation dimension
        embedding_dim: dimension of embedding
        output_dim: output dimension
    """
    def __init__(self, map_size: tuple, patch_size: tuple, obs_dim: int, embedding_dim:int, output_dim: int, activation):
        super().__init__()
        self.obs_dim = obs_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # default observation space size (height, width of binary map)
        self.H, self.W = map_size # (H, W)
        self.patch_size = patch_size # (H, W)
        # self.num_embedding = (self.H // self.patch_size[0]) * (self.W // self.patch_size[1])
        # self.pos_embeddings = nn.Embedding(num_embeddings=self.num_embedding, embedding_dim=embedding_dim)
        
        # # positional embedding encoder
        # self.embedding_encoder = nn.Sequential(
        #     nn.Linear(in_features=embedding_dim * 2, out_features=embedding_dim * 4),
        #     activation(),
        #     nn.Linear(in_features=embedding_dim * 4, out_features=embedding_dim * 4),
        #     activation(),
        #     nn.Linear(in_features=embedding_dim * 4, out_features=output_dim)
        # )
        
        # observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=64),
            activation(),
            nn.Linear(in_features=64, out_features=128),
            activation(),
            nn.Linear(in_features=128, out_features=output_dim)
        )
        
        # initializer network weight carefully..
        # self.init_weight()
        
    # def init_weight(self):
    #     torch.nn.init.trunc_normal_(self.pos_embeddings.weight.data, std=0.02) # standard vit-style choice
        
    #     def _init_linear(m: nn.Module):
    #         if isinstance(m, nn.Linear):
    #             if self.activation == nn.ReLU:
    #                 nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    #             else: # possibly Mish, GELU, SiLU ...
    #                 nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)
    #     self.embedding_encoder.apply(_init_linear)
    #     self.obs_encoder.apply(_init_linear)
        
    #     # last linear in embedding_encoder
    #     last_emb = [m for m in self.embedding_encoder.modules() if isinstance(m, nn.Linear)][-1]
    #     nn.init.zeros_(last_emb.weight)
    #     nn.init.zeros_(last_emb.bias)
        
    #     # last linear in obs_encoder
    #     last_obs = [m for m in self.obs_encoder.modules() if isinstance(m, nn.Linear)][-1]
    #     nn.init.zeros_(last_obs.weight)
    #     nn.init.zeros_(last_obs.bias)    
    
    # def calculate_pos_index(self, point):
    #     """
    #         point: Float tensor of (Start_x, Start_y, Goal_x, Goal_y)
    #     """
    #     start_x = point[:, 0]; start_y = point[:, 1]
    #     goal_x = point[:, 2]; goal_y = point[:, 3]
        
    #     s_col_id = start_x // self.patch_size[1]
    #     s_row_id = start_y // self.patch_size[0]
    #     s_pos_idx = s_col_id + s_row_id * (self.W // self.patch_size[1]) # [B, 1]
        
    #     g_col_id = goal_x // self.patch_size[1]
    #     g_row_id = goal_y // self.patch_size[0]
    #     g_pos_idx = g_col_id + g_row_id * (self.W // self.patch_size[1]) # [B, 1]
        
    #     s_pos_idx = s_pos_idx.to(torch.long) # start point index on embedding map
    #     g_pos_idx = g_pos_idx.to(torch.long) # goal point index on embedding map
    #     return s_pos_idx, g_pos_idx
        
        
    
    def forward(self, x):
        """
            x: (start_x, start_y, goal_x, goal_y) - 4 dim
        """
        # (B, output_dim)
        obs_feat =  self.obs_encoder(x)
        
        # calculate pose index
        # s_query, g_query = self.calculate_pos_index(x)
        # start_feat = self.pos_embeddings(s_query) # [B, embedding_dim]
        # goal_feat = self.pos_embeddings(g_query) # [B, embedding_dim]
        
        # start_goal_feat = torch.cat([start_feat, goal_feat], dim=1) # [B, embedding_dim * 2]
        
        # (B, output_dim)
        # embedding_feat = self.embedding_encoder(start_goal_feat)
        # output_feat = obs_feat + embedding_feat # (B, output_dim)
        output_feat = obs_feat
        
        return output_feat

def main():
    import torchinfo

    obs_encoder = OBSEmbedEncoder(map_size=(8, 8), patch_size=(2, 2), obs_dim = 4, embedding_dim = 128, output_dim = 256, activation=nn.Mish)
    obs_encoder = obs_encoder.to('cuda')

    torchinfo.summary(obs_encoder)

    data = torch.tensor([2.3, 6.7, 6.5, 2.1], device='cuda', dtype=torch.float32).unsqueeze(0)
    obs_out = obs_encoder(data)
    print("obs_out.shape: ", obs_out.shape)

if __name__ == '__main__':    
    main()
from core.comp_diffusion.cd_stgl_sml_dfu.stgl_sml_traj_planner import CompTrajPlanner # Planner network
from core.comp_diffusion.cd_stgl_sml_dfu.stgl_sml_traj_planner_v2 import CompTrajPlannerV2

from core.comp_diffusion.configs.comp_diffusion_cfg import CompDiffusionConfig # PB Diffusion Config
from core.comp_diffusion.configs.comp_diffusion_ddp_cfg import CompDiffusionDDPConfig
from core.comp_diffusion.configs.comp_diffusion_cfg_v2 import CompDiffusionConfigV2
from core.comp_diffusion.configs.comp_v2_ddp_cfg import CompDDPV2Config

from core.comp_diffusion.cd_stgl_sml_dfu.stgl_sml_diffusion_v1 import Stgl_Sml_GauDiffusion_InvDyn_V1


def build_comp_diff_from_cfg(model_id, ddp=False):
    if model_id == 0:
        if ddp:
            config = CompDiffusionDDPConfig()
        else:
            config = CompDiffusionConfig()
        config_dict = config.to_dict()
        model = CompTrajPlanner(config_dict['network_config'])
        
        diffusion_config = config_dict['diffusion_config']
        diffusion_model = Stgl_Sml_GauDiffusion_InvDyn_V1(model=model, **diffusion_config)
        
    if model_id == 1:
        if ddp:
            config = CompDDPV2Config()
        else:   
            config = CompDiffusionConfigV2()
        config_dict = config.to_dict()
        model = CompTrajPlannerV2(config_dict['network_config'])
        
        diffusion_config = config_dict['diffusion_config']
        diffusion_model = Stgl_Sml_GauDiffusion_InvDyn_V1(model=model, **diffusion_config)
        
    return diffusion_model, config_dict


from core.pb_diffusion.networks.pb_traj_planner import PBTrajPlanner # Planner network
from core.pb_diffusion.networks.pb_bil_traj_planner import PBTrajBILPlanner # Bilatent Planner network
from core.pb_diffusion.networks.pb_traj_planner_v2 import PBTrajPlannerV2 # V2 Planner network (Enhance conditioning)

from core.pb_diffusion.configs.pb_diffusion_cfg import PBDiffusionConfig # PB Diffusion Config
from core.pb_diffusion.configs.pb_diffusion_bil_cfg import PBDiffusionBILConfig # Bilatent PB Diffusion Config
from core.pb_diffusion.configs.pb_diffusion_v2_cfg import PBDiffusionV2Config # PB Diffusion Config V2
from core.pb_diffusion.configs.pb_ddp_v2_cfg import PBV2DDPConfig

from core.pb_diffusion.diffusion.diffusion_pb import GaussianDiffusionPB # Diffusion Framework

def build_pb_diff_from_cfg(model_id, ddp=False):
    if model_id == 0:
        config = PBDiffusionConfig()
        config_dict = config.to_dict()
        model = PBTrajPlanner(config_dict['network_config'])
        
        diffusion_config = config_dict['diffusion_config']
        diffusion_model = GaussianDiffusionPB(model=model, **diffusion_config)
    elif model_id == 1:
        config = PBDiffusionBILConfig()
        config_dict = config.to_dict()
        model = PBTrajBILPlanner(config_dict['network_config'])

        diffusion_config = config_dict['diffusion_config']
        diffusion_model = GaussianDiffusionPB(model=model, **diffusion_config)
    elif model_id == 2:
        if ddp:
            config = PBV2DDPConfig()
        else:
            config = PBDiffusionV2Config()
        config_dict = config.to_dict()
        model = PBTrajPlannerV2(config_dict['network_config'])
        
        diffusion_config = config_dict['diffusion_config']
        diffusion_model = GaussianDiffusionPB(model=model, **diffusion_config)
        
    return diffusion_model, config_dict


from core.rediffuser.networks.traj_planner import RediffuserPlanner
from core.rediffuser.configs.rediff_cfg import ReDiffConfig
from core.rediffuser.configs.rediff_ddp_cfg import ReDiffDDPConfig
from core.rediffuser.networks.diffuser.models.diffusion import GaussianDiffusion

def build_rediff_from_cfg(model_id, ddp=False):
    if model_id == 0:
        if ddp:
            config = ReDiffDDPConfig()
        else:
            config = ReDiffConfig()
        config_dict = config.to_dict()
        model = RediffuserPlanner(config_dict['network_config'])
        
        diffusion_config = config_dict['diffusion_config']
        diffusion_model = GaussianDiffusion(model=model, **diffusion_config)
    
    return diffusion_model, config_dict


from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)

# configuration files
from core.diffuser.config.conv1d_unet_p import Conv1dUnetTransP
from core.diffuser.config.diffuser_transformer import TransformerDecoderConfig
from core.diffuser.config.diffuser_v1 import DiffuserV1Config

from core.diffuser.config.diffuser_v2 import DiffuserV2Config
from core.diffuser.config.DDP.diffuser_ddp import DiffuserDDPConfig

from core.diffuser.config.diffuser_bil import BiLatentTrajectoryPlannerConfig
from core.diffuser.config.DDP.diffuser_bil_ddp import DiffuserBilDDPConfig

from core.diffuser.config.diffuser_mhypo import MultiHypothesisTrajectoryPlannerConfig
from core.diffuser.config.diffuser_bil_mhypo import BiLatentMultiHypoConfig
from core.diffuser.config.diffuser_mres import MultiResTrajectoryPlannerConfig
from core.diffuser.config.trajectory_planner_experimental import TrajectoryPlannerConfigExp

from core.diffuser.config.diffuser_sim_fusion import SimFusionTrajPlannerConfig
from core.diffuser.config.DDP.diffuser_sim_ddp import DiffuserSimDDPConfig

# networks
from core.diffuser.networks.embeddUnet import ConditionalUnet1D
from core.diffuser.networks.embeddUnet_p import ConditionalUnet1DTransP
from core.diffuser.networks.traj_planner.TransformerDecoder import TransformerPathGenerator
from core.diffuser.networks.traj_planner.TrajectoryPlanner import TrajectoryPlanner
from core.diffuser.networks.traj_planner.BiLatentTrajectoryPlanner import BiLatentTrajectoryPlanner
from core.diffuser.networks.traj_planner.MultiHypothesisTrajectoryPlanner import MultiHypothesisTrajectoryPlanner
from core.diffuser.networks.traj_planner.BiLatentMultiHypoPlanner import BiLatentMultiHypoPlanner
from core.diffuser.networks.traj_planner.MultiResTrajPlanner import MultiResTrajPlanner
from core.diffuser.networks.traj_planner.TrajectoryPlanner_experimental import TrajectoryPlannerExp
from core.diffuser.networks.traj_planner.SimFusionTrajPlanner import SimFusionTrajPlanner

from typing import Dict, List

def build_networks_from_config(model_id, ddp=False):
    if model_id == 0:
        config = DiffuserV1Config()
        action_dim = config.network_config['unet_config']['action_dim']
        obs_dim = config.network_config['unet_config']['action_horizon']
        obstacle_encode_dim = config.network_config['vit_config']['num_classes']
        env_encode_dim = config.network_config['mlp_config']['embed_dim']
        net = ConditionalUnet1D(input_dim=action_dim,
                          global_cond_dim=obstacle_encode_dim + env_encode_dim,
                          network_config=config.network_config, is_cnn=config.is_CNN)
    elif model_id == 1:
        config = TransformerDecoderConfig()
        net = TransformerPathGenerator(
            traj_dim=2, #noisy trajectory dimensions (=action_dim=2)
            network_config=config.network_config,
            is_cnn=config.is_CNN
        )
    elif model_id == 2:
        config = Conv1dUnetTransP()
        net = ConditionalUnet1DTransP(
                input_dim=config.network_config["unet_config"]["action_dim"],
                cond_dim=config.network_config["unet_config"]["condition_dim"],
                down_dims=config.network_config["unet_config"]["down_dims"],
                kernel_size=config.network_config["unet_config"]["kernel_size"],
                n_groups=config.network_config["unet_config"]["n_groups"],
                network_config=config.network_config,
                is_cnn=config.is_CNN
            )
    elif model_id == 3:
        if ddp:
            config = DiffuserDDPConfig()
        else:
            config = DiffuserV2Config()
        config_dict = config.to_dict()
        net_config = config_dict['network_config']
        net = TrajectoryPlanner(network_config=net_config)
    elif model_id == 4:
        if ddp:
            config = DiffuserBilDDPConfig()
        else:
            config = BiLatentTrajectoryPlannerConfig()
        config_dict = config.to_dict()
        net_config = config_dict['network_config']
        net = BiLatentTrajectoryPlanner(network_config=net_config)
    elif model_id == 5:
        config = MultiHypothesisTrajectoryPlannerConfig()
        config_dict = config.to_dict()
        net_config = config_dict['network_config']
        net = MultiHypothesisTrajectoryPlanner(network_config=net_config)
    elif model_id == 6:
        config = BiLatentMultiHypoConfig()
        config_dict = config.to_dict()
        net_config = config_dict['network_config']
        net = BiLatentMultiHypoPlanner(network_config=net_config)
    elif model_id == 7:
        config = MultiResTrajectoryPlannerConfig()
        config_dict = config.to_dict()
        net_config = config_dict['network_config']
        net = MultiResTrajPlanner(network_config=net_config)
    elif model_id == 8: # this is for experimental
        config = TrajectoryPlannerConfigExp()
        config_dict = config.to_dict()
        net_config = config_dict['network_config']
        net = TrajectoryPlannerExp(network_config=net_config)
    elif model_id == 9: # This is for simple fusion trajectory planner
        if ddp:
            config = DiffuserSimDDPConfig()
        else:
            config = SimFusionTrajPlannerConfig()
        config_dict = config.to_dict()
        net_config = config_dict['network_config']
        net = SimFusionTrajPlanner(network_config=net_config)
        
    return config, config.to_dict(), net


def build_noise_scheduler_from_config(config: Dict):
    type_noise_scheduler = config["noise_scheduler"]["type"]
    if type_noise_scheduler.lower() == "ddpm":
        return DDPMScheduler(
            num_train_timesteps=config["noise_scheduler"]["ddpm"]["num_train_timesteps"],
            beta_schedule=config["noise_scheduler"]["ddpm"]["beta_schedule"],
            clip_sample=config["noise_scheduler"]["ddpm"]["clip_sample"],
            prediction_type=config["noise_scheduler"]["ddpm"]["prediction_type"],
        )
    elif type_noise_scheduler.lower() == "ddim":
        return DDIMScheduler(
            num_train_timesteps=config["noise_scheduler"]["ddim"]["num_train_timesteps"],
            beta_schedule=config["noise_scheduler"]["ddim"]["beta_schedule"],
            clip_sample=config["noise_scheduler"]["ddim"]["clip_sample"],
            prediction_type=config["noise_scheduler"]["ddim"]["prediction_type"],
        )
    elif type_noise_scheduler.lower() == "dpmsolver":
        return DPMSolverMultistepScheduler(
            num_train_timesteps=config["noise_scheduler"]["dpmsolver"]["num_train_timesteps"],
            beta_schedule=config["noise_scheduler"]["dpmsolver"]["beta_schedule"],
            prediction_type=config["noise_scheduler"]["dpmsolver"]["prediction_type"],
            use_karras_sigmas=config["noise_scheduler"]["dpmsolver"]["use_karras_sigmas"],
        )
    else:
        raise NotImplementedError
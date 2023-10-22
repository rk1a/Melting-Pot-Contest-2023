from baselines.wrappers.meltingpot_wrapper import MeltingPotEnv
from meltingpot import substrate
from baselines.wrappers.downsamplesubstrate_wrapper import DownSamplingSubstrateWrapper
from baselines.wrappers.metapolicy_wrapper import MetaPolicyWrapper
from ml_collections import config_dict

def env_creator(env_config):
  """Build the substrate, interface with RLLIB and apply Downsampling to observations."""
  
  env_config = config_dict.ConfigDict(env_config)
  env = substrate.build(env_config['substrate'], roles=env_config['roles'])
  env = DownSamplingSubstrateWrapper(env, env_config['scaled'])
  env = MeltingPotEnv(env, env_config['shared_reward'])
  if 'meta_policy' in env_config:
    env = MetaPolicyWrapper(env, **env_config['meta_policy'])
  return env


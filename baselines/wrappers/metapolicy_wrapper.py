from immutabledict import immutabledict
from typing import List
from gymnasium import spaces
from ray.rllib.env import multi_agent_env

from meltingpot.utils.policies import policy
from meltingpot.bot import get_factory
import dm_env


PLAYER_STR_FORMAT = 'player_{index}'


class MetaPolicyWrapper(multi_agent_env.MultiAgentEnv):
  """Restricting action space to selecting from pretrained behaviors."""

  def __init__(self, env: multi_agent_env.MultiAgentEnv, behavior_names: List[str]):
    """Initializes the instance.

    Args:
      env: RLLib multi agent environment.
      behavior_names: list of names of pretrained behaviors to use.
    """
    self._env = env
    self._num_players = self._env._num_players
    self._ordered_agent_ids = self._env._ordered_agent_ids
    self._agent_ids = self._env._agent_ids
    self._n_behaviors = len(behavior_names)
    self._behavior_names = behavior_names
    self._behaviors = {}
    self._prev_states = {}
    for behavior_name in behavior_names:
        behavior = get_factory(behavior_name).build()
        self._behaviors[behavior_name] = behavior
        self._prev_states[behavior_name] = {agent_id: None for agent_id in self._ordered_agent_ids}
    
    # Meta policy chooses in a discrete action space from the list of behaviors
    self.action_space = spaces.Dict({
        agent_id: spaces.Discrete(self._n_behaviors)
        for _, agent_id in enumerate(self._env._ordered_agent_ids)
    })
    self.observation_space = self._env.observation_space

    super().__init__()

  def reset(self, *args, **kwargs):
    """Reset environment."""
    for behavior_name in self._behavior_names:
        behavior = self._behaviors[behavior_name]
        state = behavior.initial_state()
        for agent_id in self._ordered_agent_ids:
          self._prev_states[behavior_name][agent_id] = state
    return self._env.reset(*args, **kwargs)

  def step(self, action_dict):
    """Convert meta actions to actions by calling pretrained behaviors."""
    actions = {}
    prev_timestep = self._env._prev_timestep
    for idx, agent_id in enumerate(self._ordered_agent_ids):
        meta_action = action_dict[agent_id]
        # assert isinstance(meta_action, int)
        behavior_name = self._behavior_names[meta_action]
        behavior = self._behaviors[behavior_name]
        agent_timestep = dm_env.TimeStep(
           step_type=prev_timestep.step_type,
           observation=immutabledict(prev_timestep.observation[idx]),
           reward=prev_timestep.reward[idx],
           discount=prev_timestep.discount,
        )
        actions[agent_id], self._prev_states[behavior_name][agent_id] = behavior.step(agent_timestep, self._prev_states[behavior_name][agent_id])
    observations, rewards, done, done, info = self._env.step(actions)
    return observations, rewards, done, done, info

  def close(self):
    """Close environment."""
    self._env.close()
import nmmo
from typing import Dict, Type, List

from newnmmo.evaluation.team import Team
from newnmmo.scripted import baselines


class ScriptedTeam(Team):
    agent_klass: Type = None
    agents: List[baselines.Scripted]

    def __init__(
        self,
        team_id: str,
        env_config: nmmo.config.Config,
        **kwargs,
    ) -> None:
        if "policy_id" not in kwargs:
            kwargs["policy_id"] = self.agent_klass.__name__
        super().__init__(team_id, env_config, **kwargs)
        self.reset()

    def reset(self):
        assert self.agent_klass
        self.agents = [
            self.agent_klass(self.env_config, i)
            for i in range(self.env_config.TEAM_SIZE)
        ]

    def act(self, observations: Dict[int, dict]) -> Dict[int, dict]:
        actions = {i: self.agents[i](obs) for i, obs in observations.items()}
        for i in actions:
            for atn, args in actions[i].items():
                for arg, val in args.items():
                    if len(arg.edges) > 0:
                        actions[i][atn][arg] = arg.edges.index(val)
                    else:
                        targets = self.agents[i].targets
                        actions[i][atn][arg] = targets.index(val)
        return actions


class RandomTeam(ScriptedTeam):
    agent_klass = baselines.Random


class MeanderTeam(ScriptedTeam):
    agent_klass = baselines.Meander


class ForageNoExploreTeam(ScriptedTeam):
    agent_klass = baselines.ForageNoExplore


class ForageTeam(ScriptedTeam):
    agent_klass = baselines.Forage


class CombatTeam(ScriptedTeam):
    agent_klass = baselines.Combat


class CombatNoExploreTeam(ScriptedTeam):
    agent_klass = baselines.CombatNoExplore


class CombatTribridTeam(ScriptedTeam):
    agent_klass = baselines.CombatTribrid

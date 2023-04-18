from ijcai2022nmmo import CompetitionConfig, scripted, submission, RollOut
import os
from ijcai2022nmmo import Team

from torchbeast.neural_mmo.rule_agent import RuleAgent
import time

config = CompetitionConfig()
path = os.getcwd()
file_name = 'replay_'+time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) 
config.SAVE_REPLAY = os.path.join(path, 'replay', file_name)


class RuleTeam(Team):
    n_action = 5

    def __init__(self,
                 team_id: str,
                 env_config: config,
                 checkpoint_path=None):
        super().__init__(team_id, env_config)
        self.agents = [
            RuleAgent(self.env_config, i)
            for i in range(self.env_config.TEAM_SIZE)
        ]

    def act(self, observations):
        raw_observations = observations

        actions = self.transform_act(raw_observations)

        '''
        output: action{}  
        {8:{2{move:dirction,attack:{style:,target:}
        '''
        return actions

    def transform_act(self, raw_observations):
        actions = {i: self.agents[i](obs) for i, obs in raw_observations.items()}

        for i in actions:
            for atn, args in actions[i].items():
                for arg, val in args.items():
                    if len(arg.edges) > 0:
                        actions[i][atn][arg] = arg.edges.index(val)
                    else:
                        targets = self.agents[i].targets
                        actions[i][atn][arg] = targets.index(val)

        return actions


my_team = RuleTeam("Myteam", config,)

teams = []
teams.extend([scripted.CombatTeam(f"Combat-{i}", config) for i in range(3)])
teams.extend([scripted.ForageTeam(f"Forage-{i}", config) for i in range(5)])
teams.extend([scripted.RandomTeam(f"Random-{i}", config) for i in range(7)])
teams.append(my_team)

ro = RollOut(config, teams, parallel=True, show_progress=True)
ro.run(n_episode=1,render=True)

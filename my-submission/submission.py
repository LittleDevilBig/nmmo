from pathlib import Path

import torch
import torch.nn as nn
import tree
from ijcai2022nmmo import Team
from nmmo import config
import nmmo
from torchbeast.monobeast import Net, batch, unbatch
from torchbeast.neural_mmo.train_wrapper import (AttackTeam, FeatureParser,
                                                 TrainWrapper)
from torchbeast.neural_mmo.rule_agent import RuleAgent, ExploreAgent, ForageAgent

class MonobeastBaselineTeam(Team):
    n_action = 5

    def __init__(self,
                 team_id: str,
                 env_config: config.Config,
                 checkpoint_path=None):
        super().__init__(team_id, env_config)
        self.model: nn.Module = Net(None, self.n_action, False)
        if checkpoint_path is not None:
            print(f"load checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.feature_parser = FeatureParser()
        self.reset()

    def act(self, observations):
        # 8 dict
        raw_observations = observations
        observations = self.feature_parser.parse(observations)
        observations = tree.map_structure(
            lambda x: torch.from_numpy(x).view(1, 1, *x.shape), observations)
        obs_batch, ids = batch(observations,
                               self.feature_parser.feature_spec.keys())
        output, _ = self.model(obs_batch)
        output = unbatch(output, ids)
        # 动作: move {0: 4, 1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4}
        actions = {i: output[i]["action"].item() for i in output}
        # 动作:  攻击
        actions = TrainWrapper.transform_action(actions, raw_observations,
                                                self.auxiliary_script)

        return actions

    def reset(self):
        self.auxiliary_script = AttackTeam("auxiliary", self.env_config)


class RuleTeam(Team):
    n_action = 5

    def __init__(self,
                 team_id: str,
                 env_config: config.Config,
                 checkpoint_path=None):
        super().__init__(team_id, env_config)
        # rule 1 forage 2 explore 3
        self.agents = [
            RuleAgent(self.env_config, i)
            for i in range(2)
        ] + [
            ForageAgent(self.env_config, i)
            for i in range(2,4)
        ] + [
            ExploreAgent(self.env_config, i)
            for i in range(4,6)            
        ] + [
            RuleAgent(self.env_config, i)
            for i in range(6,8)
        ]
        # self.agents = [
        #     RuleAgent(self.env_config, i)
        #     for i in range(2)
        # ] + [
        #     ExploreAgent(self.env_config, i)
        #     for i in range(2,4)
        # ] + [
        #     ForageAgent(self.env_config, i)
        #     for i in range(4,6)            
        # ] + [
        #     RuleAgent(self.env_config, i)
        #     for i in range(6,8)
        # ]

    def act(self, observations):

        actions = self.transform_act(observations)

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


class Submission:
    team_klass = RuleTeam
    # init_params = {
    #     "checkpoint_path":
    #     Path(__file__).parent / "checkpoints" / "model_113016832.pt"
    # }
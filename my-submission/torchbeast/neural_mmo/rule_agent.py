from hamcrest import close_to
from newnmmo import TeamBasedEnv
from newnmmo.scripted import CombatTeam, ForageTeam, RandomTeam
from newnmmo.scripted.baselines import Scripted
from newnmmo.scripted.scripted_team import ScriptedTeam
from newnmmo.scripted import attack, move
import nmmo
from nmmo import scripting
from nmmo.lib import colors

from newnmmo.scripted import utils
import numpy as np

import random

class RuleAgent(Scripted):
    '''rule'''
    name = 'RuleBase'

    def __init__(self, config, idx):
        super().__init__(config, idx)

        self.most_value_attack = None
        self.threaten_agent_list = None
        self.ally_agent_list = None
        self.freeze_attacker_list = None
        self.threaten_player_list = None
        self.max_explore = False
        self.max_forage = False

        self.init_random_start = random.randint(15, 35)

        self.last_attack_agent_id = None
        self.last_attack_agent_max_health = None
        self.scan_agents_entity = None

        self.last_position_r = None
        self.last_position_c = None
        self.area_count = 0


    def __call__(self, obs):
        # 继承父类的参数
        super().__call__(obs)

        self.adaptive_control_and_targeting()
        self.style = nmmo.action.Mage
        self.attack()
        return self.actions

    def attack(self):
        '''Attack the current target'''
        if self.targetID is None:
            self.last_attack_agent_max_health = None

        # 换目标
        if self.targetID is not None and self.last_attack_agent_id is not None and self.targetID != self.last_attack_agent_id:
            self.last_attack_agent_max_health = nmmo.scripting.Observation.attribute(self.target, nmmo.Serialized.Entity.Health)

        self.last_attack_agent_id = self.targetID
        if self.target is not None:
            assert self.targetID is not None
            if self.last_attack_agent_max_health is None:
                self.last_attack_agent_max_health = nmmo.scripting.Observation.attribute(self.target, nmmo.Serialized.Entity.Health)
            attack.target(self.config, self.actions, self.style, self.targetID)

    def adaptive_control_and_targeting(self, explore=True):
        '''Balanced foraging, evasion, and exploration'''
        self.scan_agents()
        self.is_max_explore()
        self.is_max_forage()

        if self.attacker is not None:
            # 如果attacker的等级比自己低 或者 自己已经被冻结，则打attacker
            self.target = self.attacker    
            self.targetID = self.attackerID

            if self.emergency_forage_criterion:
                self.forage() # 该跑还是该采集，已经emergency了，肯定要采集
                return
            else:
                self.attack_target_strategy(self.attacker)
            return

        if len(self.threaten_agent_list) > 0:
            # 远离有危险目标方向
            shortestDist = np.inf
            closestAgent = None

            Entity = nmmo.Serialized.Entity
            agent = self.ob.agent

            sr = nmmo.scripting.Observation.attribute(agent, Entity.R)
            sc = nmmo.scripting.Observation.attribute(agent, Entity.C)
            start = (sr, sc)

            for target in self.threaten_agent_list:
                exists = nmmo.scripting.Observation.attribute(target, Entity.Self)
                if not exists:
                    continue
                
                tr = nmmo.scripting.Observation.attribute(target, Entity.R)
                tc = nmmo.scripting.Observation.attribute(target, Entity.C)

                goal = (tr, tc)
                dist = utils.lInfty(start, goal)

                if dist < shortestDist and dist != 0:
                    shortestDist = dist
                    closestAgent = target                

            self.target = closestAgent
            self.targetID = scripting.Observation.attribute(closestAgent, nmmo.Serialized.Entity.ID)

            if self.emergency_forage_criterion:
                self.forage() # 该跑还是该采集，已经emergency了，肯定要采集
                return
            else:
                self.attack_target_strategy(closestAgent)
            return

        self.explore_or_forage(explore)

        if self.max_explore and self.max_forage:
            is_find_path = self.go_to_center()
            if not is_find_path:
                self.explore_from_dege(ratio=2)

        if self.init_random_start > 0 and not self.emergency_forage_criterion:
            self.explore_straight()
            self.init_random_start -= 1

        if self.area_count >= 100 and not self.emergency_forage_criterion:
            is_find_path = self.go_to_center()
            if not is_find_path:
                self.explore_from_dege(ratio=2)

        # TODO 优化：判断是否有队友
        self.target_value()

    def explore_or_forage(self, explore=True):
        if self.forage_criterion or not explore:
            self.forage()
        else:
            # TODO 探索的时候在一个方向待着
            is_explore = self.explore()
            if not is_explore:
                self.forage()

    def explore_from_dege(self, ratio=1):
        '''Route away from spawn'''
        return move.explore_from_dege(self.config, self.ob, self.actions, self.spawnR,
                     self.spawnC, ratio)

    def explore_straight(self):
        return move.explore_straight(self.config, self.ob, self.actions, self.spawnR,
                     self.spawnC)

    def go_to_center(self):
        vision = self.config.NSTIM
        sz = self.config.TERRAIN_SIZE
        Entity = nmmo.Serialized.Entity
        Tile = nmmo.Serialized.Tile

        agent = self.ob.agent
        r = nmmo.scripting.Observation.attribute(agent, Entity.R)
        c = nmmo.scripting.Observation.attribute(agent, Entity.C)

        centR, centC = sz // 2, sz // 2
        vR, vC = centR - r, centC - c

        mmag = max(abs(vR), abs(vC))

        if mmag == 0:
            return False

        rr = int(np.round(vision * vR / mmag))
        cc = int(np.round(vision * vC / mmag))
        return move.pathfind(self.config, self.ob, self.actions, rr, cc)

    @property
    def forage_criterion(self) -> bool:
        '''Return true if low on food or water or forage food and water at low level'''
        # TODO dijs:
        min_level = 7
        if self.food <= min_level or self.water <= min_level:
            return True
        # 第一阶段：
        if self.food_max < 20 or self.water_max < 20:
            return True
        # 第二阶段：
        if self.food < self.food_max * 0.7 or self.water <  self.water_max * 0.7:
            return True
        # 第三阶段：
        if self.max_explore and self.max_forage:
            return False
        return self.max_explore

    @property
    def emergency_forage_criterion(self) -> bool:
        '''Return true if low on food or water'''
        min_level = 7
        return self.food <= min_level or self.water <= min_level

    def scan_agents(self):
        '''Scan the most value agent to attack and threaten agent'''
        Entity = nmmo.Serialized.Entity

        sr = nmmo.scripting.Observation.attribute(self.ob.agent, Entity.R)
        sc = nmmo.scripting.Observation.attribute(self.ob.agent, Entity.C)

        # 是否在一个位置徘徊
        if self.last_position_r is None:
            self.last_position_r = sr
            self.last_position_c = sc
            self.area_count = 0
        else:
            start = (sr, sc)
            goal = (self.last_position_r, self.last_position_c)
            dist = utils.lInfty(start, goal)
            if dist <=7:
                self.area_count += 1
            else:
                self.last_position_r = sr
                self.last_position_c = sc
                self.area_count = 0

        self.most_value_attack, self.threaten_agent_list, self.ally_agent_list = self.most_value_threaten_target(
            self.config, self.ob)

        self.attacker, self.attackerDist = attack.attacker(
            self.config, self.ob)
        
        self.most_value_ID = None
        if self.most_value_attack is not None:
            self.most_value_ID = scripting.Observation.attribute(
                self.most_value_attack, nmmo.Serialized.Entity.ID)

        self.attackerID = None
        if self.attacker is not None:
            self.attackerID = scripting.Observation.attribute(
                self.attacker, nmmo.Serialized.Entity.ID)

        self.style = None
        self.target = None
        self.targetID = None
        self.targetDist = None

    def is_max_explore(self):
        if self.max_explore:
            return

        Entity = nmmo.Serialized.Entity

        sr = nmmo.scripting.Observation.attribute(self.ob.agent, Entity.R)
        sc = nmmo.scripting.Observation.attribute(self.ob.agent, Entity.C)

        gr = self.spawnR
        gc = self.spawnC

        start = (sr, sc)
        goal = (gr, gc)
        dist = utils.lInfty(start, goal)
        if dist == 127:
            self.max_explore = True

    def is_max_forage(self):
        if self.max_forage:
            return
        if (self.food_max + self.water_max)/2 > 50:
            self.max_forage = True

    def most_value_threaten_target(self, config, ob):

        Entity = nmmo.Serialized.Entity
        agent = ob.agent
        self_population = scripting.Observation.attribute(agent, Entity.Population)
        selfLevel = scripting.Observation.attribute(agent, Entity.Level)

        most_value_npc = None
        most_value_player = None
        threaten_target_list = []
        ally_list = []
        self.scan_agents_entity = {}
        most_value_npc_level = 0
        most_value_player_level = 0

        for target in ob.agents:
            exists = nmmo.scripting.Observation.attribute(target, Entity.Self)
            if not exists:
                continue
            
            population = nmmo.scripting.Observation.attribute(target, Entity.Population)
            targLevel = nmmo.scripting.Observation.attribute(target, Entity.Level)
            target_id = nmmo.scripting.Observation.attribute(target, Entity.ID)
            
            self.scan_agents_entity[target_id] = target

            if population >= 0 and population != self_population:
                if targLevel <= selfLevel <= 5 or selfLevel >= targLevel + 3:
                    if most_value_player_level < targLevel:
                        most_value_player = target
                        most_value_player_level = targLevel
                else:
                    threaten_target_list.append(target)

            elif population == self_population:
                ally_list.append(target)
                    
            elif population == -1:
                if most_value_npc_level < targLevel: 
                    most_value_npc = target
                    most_value_npc_level = targLevel

            elif population < -1:
                # TODO 细化，通过生命值等信息判断是否要打
                if targLevel <= selfLevel <= 5 or selfLevel >= targLevel + 3:
                    if most_value_npc_level < targLevel: 
                        most_value_npc = target
                        most_value_npc_level = targLevel
                elif population == -3:
                    threaten_target_list.append(target)    
            
        if most_value_player is not None:
            return most_value_player, threaten_target_list, ally_list
        elif most_value_npc is not None:
            return most_value_npc, threaten_target_list, ally_list
        else:
            return None, threaten_target_list, ally_list

    def target_value(self):
        '''Target the most value agent'''
        if self.most_value_attack is None:
            return False

        # 判断对手类型，如果是P直接打，如果对方血量和等级都比自己低，直接打
        self.attack_target_strategy(self.most_value_attack)        

    def attack_target_strategy(self, target_agent=None):
        if target_agent is None:
            return False
        
        Entity = nmmo.Serialized.Entity

        target_id = nmmo.scripting.Observation.attribute(target_agent, Entity.ID)

        if self.last_attack_agent_id is not None and target_id != self.last_attack_agent_id:
            # last attack agent 还活着
            if self.last_attack_agent_id in self.scan_agents_entity.keys():
                last_attack_target_health = nmmo.scripting.Observation.attribute(self.scan_agents_entity[self.last_attack_agent_id], Entity.Health)
                if last_attack_target_health < 0.7 * self.last_attack_agent_max_health:
                    target_agent = self.scan_agents_entity[self.last_attack_agent_id]

        self_level = nmmo.scripting.Observation.attribute(self.ob.agent, Entity.Level)
        target_level = nmmo.scripting.Observation.attribute(target_agent, Entity.Level)

        self_health = nmmo.scripting.Observation.attribute(self.ob.agent, Entity.Health)
        target_health = nmmo.scripting.Observation.attribute(target_agent, Entity.Health)
        target_population = nmmo.scripting.Observation.attribute(target_agent, Entity.Population)

        if target_population == -1:
            self.attack_target_direct(target_agent)
            return
        
        level_satisfy = target_level <= self_level <= 5 or self_level >= target_level + 3
        health_satisfy = self_health > target_health

        if level_satisfy and health_satisfy:
            self.attack_target_direct(target_agent)
            return
        
        if not level_satisfy and not health_satisfy and not self.emergency_forage_criterion:
            self.evade_from_target(target_agent)
            self.target = target_agent
            self.targetID = scripting.Observation.attribute(
                    target_agent, nmmo.Serialized.Entity.ID)
            return
        
        if not health_satisfy and level_satisfy and self_health <= 10 and not self.emergency_forage_criterion:
            self.evade_from_target(target_agent)
            self.target = target_agent
            self.targetID = scripting.Observation.attribute(
                    target_agent, nmmo.Serialized.Entity.ID)
            return

        self.attack_target_pull(target_agent)


    def attack_target_direct(self, target_agent=None):
        if target_agent is None:
            return False
        Entity = nmmo.Serialized.Entity

        sr = nmmo.scripting.Observation.attribute(self.ob.agent, Entity.R)
        sc = nmmo.scripting.Observation.attribute(self.ob.agent, Entity.C)

        gr = nmmo.scripting.Observation.attribute(target_agent, Entity.R)
        gc = nmmo.scripting.Observation.attribute(target_agent, Entity.C)
        target_population = nmmo.scripting.Observation.attribute(target_agent, Entity.Population)

        start = (sr, sc)
        goal = (gr, gc)
        dist = utils.lInfty(start, goal)
        
        if target_population == -1:
            if dist >= 4 and not self.emergency_forage_criterion:
                self.close_to_target(target_agent)
        else:
            if dist > 4 and not self.emergency_forage_criterion:
                self.close_to_target(target_agent)  

        self.target = target_agent
        self.targetID = scripting.Observation.attribute(
                target_agent, nmmo.Serialized.Entity.ID)        


    def attack_target_pull(self, target_agent=None):
        if target_agent is None:
            return False
        Entity = nmmo.Serialized.Entity

        sr = nmmo.scripting.Observation.attribute(self.ob.agent, Entity.R)
        sc = nmmo.scripting.Observation.attribute(self.ob.agent, Entity.C)

        gr = nmmo.scripting.Observation.attribute(target_agent, Entity.R)
        gc = nmmo.scripting.Observation.attribute(target_agent, Entity.C)

        start = (sr, sc)
        goal = (gr, gc)
        dist = utils.lInfty(start, goal)
        # TODO 拉扯距离把控 121的423步
        if dist > 4 and not self.emergency_forage_criterion:
            self.close_to_target(target_agent)
        elif dist < 4 and not self.emergency_forage_criterion:
            self.evade_from_target(target_agent)

        self.target = target_agent
        self.targetID = scripting.Observation.attribute(
                target_agent, nmmo.Serialized.Entity.ID)

    def close_to_target(self, target_agent):
        move.colse_target(self.config, self.ob, self.actions, target_agent)

    def evade_from_target(self, target_agent):
        move.evade(self.config, self.ob, self.actions, target_agent)

    def target_weak(self):
        '''Target the nearest agent if it is weak'''
        if self.closest is None:
            return False

        selfLevel = scripting.Observation.attribute(
            self.ob.agent, nmmo.Serialized.Entity.Level)
        targLevel = scripting.Observation.attribute(
            self.closest, nmmo.Serialized.Entity.Level)

        if targLevel <= selfLevel <= 5 or selfLevel >= targLevel + 3:
            self.target = self.closest
            self.targetID = self.closestID
            self.targetDist = self.closestDist


class ExploreAgent(RuleAgent):
    '''rule'''
    name = 'explore'

    def __init__(self, config, idx):
        super().__init__(config, idx)

    def explore_or_forage(self, explore=True):
        if self.forage_criterion or not explore:
            self.forage()
        else:
            is_explore = self.explore_from_dege()
            if not is_explore:
                self.forage()

    @property
    def forage_criterion(self) -> bool:
        '''Return true if low on food or water or forage food and water at low level'''
        min_level = 7
        if self.food <= min_level or self.water <= min_level:
            return True
        # 第一阶段：
        if self.food_max < 15 or self.water_max < 15:
            return True
        # 第二阶段：
        if self.food < self.food_max * 0.6 or self.water <  self.water_max * 0.6:
            return True

        return self.max_explore


class ForageAgent(RuleAgent):
    '''rule'''
    name = 'forage'

    def __init__(self, config, idx):
        super().__init__(config, idx)

    def explore_or_forage(self, explore=True):
        if self.forage_criterion or not explore:
            self.forage()
        else:
            is_explore = self.explore_from_dege(ratio=2)
            if not is_explore:
                self.forage()

    @property
    def forage_criterion(self) -> bool:
        '''Return true if low on food or water or forage food and water at low level'''
        return self.max_forage

    @property
    def forage_criterion(self) -> bool:
        '''Return true if low on food or water or forage food and water at low level'''
        min_level = 7
        if self.food <= min_level or self.water <= min_level:
            return True
        # 第一阶段：
        if self.food_max < 20 or self.water_max < 20:
            return True
        # 第二阶段：
        if self.food < self.food_max * 0.8 or self.water <  self.water_max * 0.8:
            return True

        return self.max_explore


class CombatAgent(RuleAgent):
    '''rule'''
    name = 'combat'

    def __init__(self, config, idx):
        super().__init__(config, idx)
    

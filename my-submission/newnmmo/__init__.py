from newnmmo.env.team_based_env import TeamBasedEnv
from newnmmo.evaluation.team import Team
from newnmmo.evaluation.rollout import RollOut
from newnmmo.config import CompetitionConfig
from newnmmo.evaluation.proxy import ProxyTeam
from newnmmo.evaluation.proxy import TeamServer
from newnmmo.env.metrics import Metrics
from newnmmo.env.stat import Stat
from newnmmo.timer import timer
from newnmmo.evaluation.rating import RatingSystem
from newnmmo.evaluation.analyzer import TeamResult
from newnmmo.evaluation import analyzer
from newnmmo import exception

__all__ = [
    "TeamBasedEnv",
    "Team",
    "RollOut",
    "CompetitionConfig",
    "ProxyTeam",
    "TeamServer",
    "Metrics",
    "Stat",
    "timer",
    "RatingSystem",
    "TeamResult",
    "analyzer",
    "exception",
]

from newnmmo.version import version

__version__ = version

# monkey patch nmmo
from nmmo.entity import Player


def _packet(self):
    data = super(Player, self).packet()

    data["entID"] = self.entID
    data["annID"] = self.population

    data["base"] = self.base.packet()
    data["resource"] = self.resources.packet()
    data["skills"] = self.skills.packet()

    data["metrics"] = {
        "PlayerDefeats": self.history.playerKills,
        "Equipment": self.loadout.defense,
        "Exploration": self.history.exploration,
        "Foraging":
        (self.skills.fishing.level + self.skills.hunting.level) / 2.0,
        "Achievement": self.diary.cumulative_reward,
        "TimeAlive": self.history.timeAlive.val,
    }

    return data


def _monkey_patch():
    Player.packet = _packet


_monkey_patch()

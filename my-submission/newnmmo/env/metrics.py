import nmmo
from nmmo import entity
from typing import List, Sequence, Callable

from newnmmo import tasks


class Metrics(dict):
    @classmethod
    def names(cls) -> List[str]:
        return [
            "PlayerDefeats",
            "Equipment",
            "Exploration",
            "Foraging",
            "TimeAlive",
        ]

    @classmethod
    def collect(cls, env: nmmo.Env, player: entity.Player) -> "Metrics":
        realm = env.realm
        return Metrics(
            **{
                "PlayerDefeats": float(tasks.player_kills(realm, player)),
                "Equipment": float(tasks.equipment(realm, player)),
                "Exploration": float(tasks.exploration(realm, player)),
                "Foraging": float(tasks.foraging(realm, player)),
                "TimeAlive": float(player.history.timeAlive.val),
            })

    @classmethod
    def sum(cls, metrices: Sequence["Metrics"]) -> "Metrics":
        return cls.reduce(sum, metrices)

    @classmethod
    def max(cls, metrices: Sequence["Metrics"]) -> "Metrics":
        return cls.reduce(max, metrices)

    @classmethod
    def min(cls, metrices: Sequence["Metrics"]) -> "Metrics":
        return cls.reduce(min, metrices)

    @classmethod
    def avg(cls, metrices: Sequence["Metrics"]) -> "Metrics":
        return cls.reduce(lambda x: sum(x) / len(x) if len(x) else 0, metrices)

    @classmethod
    def reduce(cls, func: Callable,
               metrices: Sequence["Metrics"]) -> "Metrics":
        names = cls.names()
        values = [[m[name] for m in metrices] for name in names]
        return Metrics(**dict(zip(names, list(map(func, values)))))

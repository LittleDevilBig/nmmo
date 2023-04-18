import nmmo

from newnmmo.tasks import All


class CompetitionConfig(nmmo.config.Medium, nmmo.config.AllGameSystems):
    NPOP = 16
    NENT = 8 * 16

    @property
    def SPAWN(self):
        return self.SPAWN_CONCURRENT

    AGENT_LOADER = nmmo.config.TeamLoader

    AGENTS = NPOP * [nmmo.Agent]

    TASKS = All

    NMAPS = 40

    PATH_MAPS = "maps"

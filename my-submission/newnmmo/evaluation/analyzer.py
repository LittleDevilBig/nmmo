import numpy as np
from typing import List, Dict
from collections import defaultdict

from newnmmo.env.stat import Stat
from newnmmo.env.metrics import Metrics


class TeamResult:
    policy_id: str
    stat: Stat
    n_timeout: int = 0

    def __init__(self, policy_id: str, stat: Stat, n_timeout: int = 0) -> None:
        self.policy_id = policy_id
        self.stat = stat
        self.n_timeout = n_timeout
        self.achievement = self._achievement()

    def _achievement(self):
        a = 0

        high = 21
        mid = 10
        low = 4

        player_defeats = self.stat["max"]["PlayerDefeats"]
        if player_defeats >= 6:
            a += high
        elif player_defeats >= 3:
            a += mid
        elif player_defeats >= 1:
            a += low

        equipment = self.stat["max"]["Equipment"]
        if equipment >= 20:
            a += high
        elif equipment >= 10:
            a += mid
        elif equipment >= 1:
            a += low

        exploration = self.stat["max"]["Exploration"]
        if exploration >= 127:
            a += high
        elif exploration >= 64:
            a += mid
        elif exploration >= 32:
            a += low

        foraging = self.stat["max"]["Foraging"]
        if foraging >= 50:
            a += high
        elif foraging >= 35:
            a += mid
        elif foraging >= 20:
            a += low

        return a


def avg_results(results: List[Dict[int, TeamResult]]) -> Dict[int, TeamResult]:
    if not results:
        return {}

    all_results: Dict[int, List[TeamResult]] = {}
    for result_by_team in results:
        for team_idx, result in result_by_team.items():
            if team_idx not in all_results:
                all_results[team_idx] = []
            all_results[team_idx].append(result)
    avg_result_by_team: Dict[int, TeamResult] = {}
    for team_idx in all_results:
        ss = {
            key: [result.stat[key] for result in all_results[team_idx]]
            for key in Stat.names()
        }
        avg_result_by_team[team_idx] = TeamResult(
            all_results[team_idx][0].policy_id,
            Stat(**{key: Metrics.avg(ss[key])
                    for key in Stat.names()}),
        )
        avg_result_by_team[team_idx].achievement = np.mean(
            [r.achievement for r in all_results[team_idx]])
    return avg_result_by_team


def topn_team_inds(results: List[Dict[int, TeamResult]],
                   n=1) -> List[List[int]]:
    if not results:
        return []

    r = []
    for result_by_team in results:
        values = [result.achievement for result in result_by_team.values()]
        values = sorted(list(set(values)), reverse=True)
        topn = []
        for v in values:
            if len(topn) >= n:
                break
            topn.extend([
                i for i in result_by_team if result_by_team[i].achievement == v
            ])
        r.append(topn)

    return r


def topn_counts(results: List[Dict[int, TeamResult]], n=1) -> Dict[int, int]:
    if not results:
        return {}

    r = {i: 0 for i in results[0]}
    topn_inds = topn_team_inds(results, n)
    for topn in topn_inds:
        for i in topn:
            r[i] += 1
    return r


def topn_probs(results: List[Dict[int, TeamResult]], n=1) -> Dict[int, float]:
    if not results:
        return {}

    topn_cnts = topn_counts(results, n)
    return {i: cnt / len(results) for i, cnt in topn_cnts.items()}


def topn_count_by_policy(results: List[Dict[int, TeamResult]],
                         n=1) -> Dict[str, int]:
    if not results:
        return {}

    r = {v.policy_id: 0 for v in results[0].values()}
    topn_inds = topn_team_inds(results, n)
    for rind, topn in enumerate(topn_inds):
        policy_ids = list(set([results[rind][tind].policy_id
                               for tind in topn]))
        for p in policy_ids:
            r[p] += 1

    return r


def topn_prob_by_policy(results: List[Dict[int, TeamResult]],
                        n=1) -> Dict[str, float]:
    if not results:
        return {}

    topn_cnts = topn_count_by_policy(results, n)
    return {p: cnt / len(results) for p, cnt in topn_cnts.items()}


def avg_stat_by_policy(
        result_by_team: Dict[int, TeamResult]) -> Dict[str, Stat]:
    d = defaultdict(lambda: [])
    for result in result_by_team.values():
        d[result.policy_id].append(result)

    ret = {}
    for policy_id, results in d.items():
        ret[policy_id] = Stat(
            **{
                key: Metrics.avg([r.stat[key] for r in results])
                for key in Stat.names()
            })

    return ret


def achievement_by_policy(
        result_by_team: Dict[int, TeamResult]) -> Dict[str, float]:
    d = defaultdict(lambda: [])
    for result in result_by_team.values():
        d[result.policy_id].append(result)

    ret = {}
    for policy_id, results in d.items():
        ret[policy_id] = np.mean([r.achievement for r in results])

    return ret


def n_timeout(results: List[Dict[int, TeamResult]]) -> Dict[str, int]:
    d = defaultdict(lambda: 0)
    for result_by_team in results:
        for result in result_by_team.values():
            d[result.policy_id] += result.n_timeout
    return d

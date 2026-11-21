"""
Episode planning and measurement.

plan_episode   runs a policy over a trace and records, for every window, the action,
               the configuration used for execution (assignment, threads), the
               adaptation overheads and the cost-model prediction.
measure_plans  measures the execution time of every planned window on the real
               executor. Policies are interleaved per window in a random order so
               that machine-state drift affects all of them equally.
"""

import time

import numpy as np

from runtime.bsp_executor import build_layout
from runtime.environment import ACTION_NAMES


def plan_episode(rt, trace, policy, keep_obs=False):
    state = rt.initial_state()
    records = []
    for t, w in enumerate(trace):
        state = rt.ingest(state, w)
        t_obs = time.perf_counter()
        obs = rt.observe(state, w)
        t_obs = time.perf_counter() - t_obs
        action = policy.decide(rt, state, obs, trace, t)
        # policies that read the runtime state also pay for computing it
        controller_time = policy.controller_time + (t_obs if getattr(policy, "uses_state", False) else 0.0)
        new_state, over = rt.apply(state, w, action)
        rec = {
            "t": t,
            "regime": w.regime,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "assign": new_state.assign,
            "tau": new_state.tau.copy(),
            "adapt_s": over["adapt"],
            "migration_s": over["migration"],
            "migrated": over["migrated"],
            "controller_s": controller_time,
            "predicted_exec_s": rt.predict(new_state, w),
            "info": obs.info,
        }
        if keep_obs:
            rec["obs"] = obs
        if getattr(policy, "last_costs", None) is not None:
            rec["oracle_costs"] = dict(policy.last_costs)
            rec["label"] = int(min(policy.last_costs, key=policy.last_costs.get))
        records.append(rec)
        state = rt.end_window(new_state, w)
    return records


def measure_plans(rt, trace, plans: dict, reps: int, seed: int = 0):
    """plans: name -> records. Adds 'exec_s' (list of reps) to every record."""
    rng = np.random.default_rng(seed)
    names = list(plans)
    k = rt.cfg.k
    for t, w in enumerate(trace):
        layouts = {name: build_layout(w.g, plans[name][t]["assign"], k) for name in names}
        for name in names:
            plans[name][t]["exec_s"] = []
            plans[name][t]["compute_s"] = []
            plans[name][t]["exchange_s"] = []
        for _ in range(reps):
            for name in rng.permutation(names).tolist():
                rec = plans[name][t]
                out = rt.executor.run(w.g, rec["assign"], k, w.active, rec["tau"], layout=layouts[name])
                rec["exec_s"].append(out["time_exec"])
                rec["compute_s"].append(out["time_compute"])
                rec["exchange_s"].append(out["time_exchange"])
    return plans

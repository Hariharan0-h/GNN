"""
Adaptation policies compared in the dynamic experiments.

Every policy maps (runtime, state, observation, trace, t) to an action:
  0 NO_ACTION, 1 THREAD_ADJUST, 2 REBALANCE (ACGR, local repair),
  3 REPARTITION (METIS from scratch, relabelled; global repartitioning)
"""

import time

import numpy as np
import torch

from runtime.environment import NO_ACTION, REBALANCE, REPARTITION, THREAD_ADJUST, lookahead_costs


class Policy:
    name = "policy"
    controller_time = 0.0  # seconds spent deciding in the last call (counted as overhead)

    def decide(self, rt, state, obs, trace, t) -> int:
        raise NotImplementedError


class ConstantPolicy(Policy):
    def __init__(self, action, name):
        self.action = action
        self.name = name

    def decide(self, rt, state, obs, trace, t):
        self.controller_time = 0.0
        return self.action


def static_policy():
    return ConstantPolicy(NO_ACTION, "Static")


def always_thread_adjust():
    return ConstantPolicy(THREAD_ADJUST, "Always-TA")


def always_rebalance():
    return ConstantPolicy(REBALANCE, "Always-ACGR")


class PeriodicMETIS(Policy):
    """Repartition from scratch with METIS every `period` windows (scratch-remap)."""

    def __init__(self, period=5):
        self.period = period
        self.name = f"Periodic-METIS({period})"

    def decide(self, rt, state, obs, trace, t):
        self.controller_time = 0.0
        return REPARTITION if (t + 1) % self.period == 0 else NO_ACTION


class ThresholdPolicy(Policy):
    """
    Rule-based trigger on the same features the learned controllers see:
      max cut drift > drift_th and communication fraction > comm_th -> REPARTITION
      max cut drift > drift_th                                       -> REBALANCE
      max thread pressure > pressure_th                              -> THREAD_ADJUST
      otherwise                                                         NO_ACTION
    Thresholds are tuned on the training samples (gnn/train_controller.py).
    """

    def __init__(self, drift_th=0.05, pressure_th=1.5, comm_th=float("inf")):
        self.drift_th = drift_th
        self.pressure_th = pressure_th
        self.comm_th = comm_th
        self.name = "Threshold"
        self.uses_state = True

    @staticmethod
    def rule(drift, pressure, comm, drift_th, pressure_th, comm_th):
        """Vectorised rule: arrays of max drift, max pressure, comm fraction -> actions."""
        drift, pressure, comm = np.asarray(drift), np.asarray(pressure), np.asarray(comm)
        out = np.full(drift.shape, NO_ACTION)
        out[pressure > pressure_th] = THREAD_ADJUST
        out[drift > drift_th] = REBALANCE
        out[(drift > drift_th) & (comm > comm_th)] = REPARTITION
        return out

    def decide(self, rt, state, obs, trace, t):
        start = time.perf_counter()
        action = int(self.rule(obs.x[:, 4].max(), obs.x[:, 1].max(), obs.globals[1],
                               self.drift_th, self.pressure_th, self.comm_th))
        self.controller_time = time.perf_counter() - start
        return action


class LearnedPolicy(Policy):
    def __init__(self, model, name):
        from gnn.controller_model import observation_to_data

        self.model = model.eval()
        self.name = name
        self.uses_state = True
        self._to_data = observation_to_data

    @torch.no_grad()
    def decide(self, rt, state, obs, trace, t):
        start = time.perf_counter()
        data = self._to_data(obs)
        batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        logits = self.model(data.x, data.edge_index, batch)
        action = int(torch.argmax(logits, dim=1).item())
        self.controller_time = time.perf_counter() - start
        return action


class MPCPolicy(Policy):
    """
    Deployable model-based baseline (model-predictive control with a persistence forecast).
    It evaluates all actions with the calibrated cost model, assuming the current window
    repeats for `horizon` windows. Its decision time, which includes the speculative ACGR
    and METIS runs needed to evaluate REBALANCE and REPARTITION, is charged as overhead.
    """

    uses_state = True

    def __init__(self, horizon=8):
        self.horizon = horizon
        self.name = "MPC"

    def decide(self, rt, state, obs, trace, t):
        start = time.perf_counter()
        costs = lookahead_costs(rt, state, [trace[t]] * self.horizon, 0, self.horizon)
        self.controller_time = time.perf_counter() - start
        return int(min(costs, key=costs.get))


class OraclePolicy(Policy):
    """
    Look-ahead oracle (not deployable: it sees the next `horizon` windows).
    Picks the action with the lowest predicted cost; used for labels and as an upper bound.
    """

    def __init__(self, horizon=8):
        self.horizon = horizon
        self.name = "Oracle"
        self.last_costs = None

    def decide(self, rt, state, obs, trace, t):
        self.last_costs = lookahead_costs(rt, state, trace, t, self.horizon)
        self.controller_time = 0.0
        return int(min(self.last_costs, key=self.last_costs.get))


class EpsilonOracle(OraclePolicy):
    """Behaviour policy for data collection: oracle action with prob 1 - eps, else random."""

    def __init__(self, horizon=8, eps=0.3, seed=0):
        super().__init__(horizon)
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        self.name = "EpsOracle"
        self.last_label = None

    def decide(self, rt, state, obs, trace, t):
        best = super().decide(rt, state, obs, trace, t)
        self.last_label = best
        if self.rng.random() < self.eps:
            return int(self.rng.integers(4))
        return best

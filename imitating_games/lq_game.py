import numpy as np

from dataclasses import dataclass
from typing import Iterable, List

@dataclass
class AffineStrategy:
    P: np.ndarray
    a: np.ndarray

    def control_input(self, x: np.ndarray):
        return -self.P @ x - self.a


@dataclass
class LinearDynamics:
    """
    Linear dynamics with system matrix `A` and input matrix `B` in discrete time.
    """

    A: np.ndarray
    B: np.ndarray

    @property
    def dims(self):
        return self.B.shape

    def next_state(self, x: np.ndarray, u: np.ndarray):
        return self.A @ x + self.B @ u

    def rollout(self, stage_strategies: Iterable[AffineStrategy], x0: np.ndarray):
        """
        Simulates the dynamical system forward in time by choosing controls according to
        `stage_strategies` starting from initial state `x0`.
        """

        trajectory = [x0]

        for strategy in stage_strategies:
            x = trajectory[-1]
            u = strategy.control_input(x)
            trajectory.append(self.next_state(x, u))

        return trajectory


@dataclass
class QuadraticCost:
    """
    A simple wrapper for a quadratic cost primitive that maps a vector `x` to a scalar cost:
    x.T * Q * x + 2*x.T * l.
    """

    Q: np.ndarray
    l: np.ndarray

    def __call__(self, x: np.ndarray):
        return 0.5 * x.T @ self.Q @ x + self.l.T @ x


@dataclass
class Player:
    """
    A player in a dynamic game.
    """

    "The indices of this player's position in the joint state vector `x`."
    position_indices: List[int]
    "The indices of this players' inputs in the joint input vector `u`."
    input_indices: List[int]
    state_cost: QuadraticCost
    input_cost: QuadraticCost
    "This players color for visualization"
    viz_color: str = "black"


@dataclass
class LQFeedbackGame:
    dynamics: LinearDynamics
    players: List[Player]
    horizon: int

    def visualize(self, trajectory: List[np.ndarray], **args):
        raise NotImplementedError

    def solve(self):
        """
        Solves a game with linear `dynamics` over a collection of `players` for a fixed `horizon` to a
        feedback Nash equilibrium. Each player has time-separable quadratic state and input cost.

        Returns a *joint* `AffineStrategy` that can be applied to the joint `dynamics`.

        The equations closely follow the derivation of Corollary 6.1 in Chapter 6.2.2 of
            Başar, Tamer, and Geert Jan Olsder. Dynamic noncooperative game theory. Society for
            Industrial and Applied Mathematics, 1998.

        The code has been ported from this Julia implementation:
        https://github.com/lassepe/iLQGames.jl/blob/master/src/solve_lq_game.jl
        """
        nx, nu = self.dynamics.dims
        cost2go = [player.state_cost for player in self.players]

        S = np.zeros((nu, nu))
        YP = np.zeros((nu, nx))
        Ya = np.zeros(nu)

        stage_strategies = []

        for _ in range(self.horizon):
            for ii, p in enumerate(self.players):
                BiZi = self.dynamics.B[:, p.input_indices].T @ cost2go[ii].Q
                S[p.input_indices, :] = (
                    p.input_cost.Q[p.input_indices, :] + BiZi @ self.dynamics.B
                )
                YP[p.input_indices, :] = BiZi @ self.dynamics.A
                Ya[p.input_indices] = (
                    self.dynamics.B[:, p.input_indices].T @ cost2go[ii].l
                    + p.input_cost.l[p.input_indices]
                )

            # compute strategy for this stage
            Sinv = np.linalg.inv(S)
            P = Sinv @ YP
            a = Sinv @ Ya

            # Update the cost2go
            F = self.dynamics.A - self.dynamics.B @ P
            b = -self.dynamics.B @ a
            for ii, p in enumerate(self.players):
                PRi = P.T @ p.input_cost.Q
                z = (
                    F.T @ (cost2go[ii].l + cost2go[ii].Q @ b)
                    + p.state_cost.l
                    + PRi @ a
                    - P.T @ p.input_cost.l
                )
                Z = F.T @ cost2go[ii].Q @ F + p.state_cost.Q + PRi @ P
                cost2go[ii] = QuadraticCost(Z, z)

            # This is the joint strategy for both players
            stage_strategies.insert(0, AffineStrategy(P, a))

        return stage_strategies

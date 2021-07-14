import numpy as np
import altair as alt
import operator

from functools import reduce
from typing import Iterable
from typing import List
from scipy.linalg import block_diag

from . import lq_game as lqg


def _pointmass2d(dt: float):
    """
    Constructs a discrete time 2D-pointmass with discretization time step `dt`.
    """
    A = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    B = np.array([[0, 0], [0, 0], [dt, 0], [0, dt]])

    return lqg.LinearDynamics(A, B)


def _product_system(subsystems: Iterable[lqg.LinearDynamics]):
    """
    Creates the product dynamical system from a list of subsystems.
    """
    A = block_diag(*(s.A for s in subsystems))
    B = block_diag(*(s.B for s in subsystems))
    return lqg.LinearDynamics(A, B)


class GuidanceFeedbackGame(lqg.LQFeedbackGame):
    """
    A guidance game formulated over two players (P1, P2). P2 wants to be close to P1. P1 wants P2 to
    reach reach a specific goal location while mainting a low velocity. All players want to minimize
    their own control effort.

    - `dt`:                 time discretization
    - `horizon`:            number of time-steps of the game
    - `goal_position`:      The goal position that P1 wants P2 to reach.
    - `input_cost_p1`:      Scales P1's input cost.
    - `vel_cost_p1`:        Scales P1's cost for P2's velocity.
    - `goal_cost_p1`:       Scales P1's cost for P2's deviciation from the `goal_position`.
    - `input_cost_p2`:      Scales P2's input cost.
    - `tracking_cost_p2`:   Scales P2's cost for being far from P1.
    """

    def __init__(
        self,
        dt=0.1,
        horizon=200,
        goal_position=np.array([-0, -1]),
        input_cost_p1=1,
        vel_cost_p1=10,
        goal_cost_p1=1,
        input_cost_p2=2,
        tracking_cost_p2=1,
    ):

        n_players = 2
        dynamics = _product_system([_pointmass2d(dt) for _ in range(n_players)])
        self.goal_position = goal_position

        p1 = lqg.Player(
            viz_color="red",
            position_indices=[0, 1],
            input_indices=[0, 1],
            # P1 wants P2 to reach the `goal_position` with minimal velocity.
            state_cost=lqg.QuadraticCost(
                Q=block_diag(
                    np.zeros((4, 4)), goal_cost_p1 * np.eye(2), vel_cost_p1 * np.eye(2)
                ),
                l=np.concatenate([np.zeros(4), -goal_position, np.zeros(2)]),
            ),
            input_cost=lqg.QuadraticCost(
                Q=input_cost_p1 * block_diag(np.eye(2), np.zeros((2, 2))),
                l=input_cost_p1 * np.zeros(4),
            ),
        )

        qblock_p2 = block_diag(np.eye(2), np.zeros((2, 2)))
        p2 = lqg.Player(
            viz_color="blue",
            position_indices=[4, 5],
            input_indices=[2, 3],
            # P2 wants to minimize the distance to P1
            state_cost=lqg.QuadraticCost(
                Q=tracking_cost_p2
                * np.block(
                    [
                        [qblock_p2, -qblock_p2],
                        [-qblock_p2, qblock_p2],
                    ]
                ),
                l=tracking_cost_p2 * np.zeros(8),
            ),
            input_cost=lqg.QuadraticCost(
                Q=input_cost_p2 * block_diag(np.zeros((2, 2)), np.eye(2)),
                l=input_cost_p2 * np.zeros(4),
            ),
        )
        players = [p1, p2]
        super().__init__(dynamics, players, horizon)

    def _visualize_player(self, trajectory, p, selector, title):
        data = alt.Data(
            values=[
                {
                    "px": s[p.position_indices[0]],
                    "py": s[p.position_indices[1]],
                    "t": t,
                }
                for t, s in enumerate(trajectory)
            ]
        )

        return (
            alt.Chart(data)
            .mark_line(point=True, tooltip={"content": "data"}, clip=True)
            .encode(
                x=alt.X("px:Q", scale={"domain": [-2, 2]}),
                y=alt.Y("py:Q", scale={"domain": [-2, 2]}),
                order="t:Q",
                opacity=alt.condition(
                    alt.datum.t <= selector.t, alt.value(1), alt.value(0)
                ),
                # opacity=alt.datum.t / selector.t
                color=alt.condition(
                    alt.datum.t >= selector.t,
                    alt.value(p.viz_color),
                    alt.value("light gray"),
                ),
            )
            .properties(title=title)
        )

    def visualize(self, trajectory: List[np.ndarray]):
        title = f"Blue wants to be close to Red. Red wants Blue to reach the {self.goal_position} position with minimal velocity."
        slider = alt.binding_range(
            min=1, max=len(trajectory) - 1, step=1, name="time step:"
        )
        selector = alt.selection_single(fields=["t"], bind=slider, init={"t": 1})
        return reduce(
            operator.add,
            (
                self._visualize_player(trajectory, p, selector, title)
                for p in self.players
            ),
        ).add_selection(selector)


def demo(x0=np.array([1.5, 0, 0, 0, -1.5, 0.0, 0, 0])):
    g = GuidanceFeedbackGame()
    stage_strategies = g.solve()

    # TODO: move one level up
    trajectory = g.dynamics.rollout(
        stage_strategies,
        x0,
    )

    viz = g.visualize(
        trajectory,
    )

    viz.properties(width=500, height=500).show()

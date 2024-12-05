import jax
import numpy as np
import jax.numpy as jnp
from abc import ABC, abstractmethod
from tqdm import tqdm


# define dynamics of bicycle model
def step(state, control):
    # state: [x, y, theta, v]
    # control: [delta, a]

    # parameters
    L = 1.0  # length of vehicle
    dt = 0.01  # time step

    # state
    x, y, theta, v = state
    delta, a = control

    # update state
    x += v * jnp.cos(theta) * dt
    y += v * jnp.sin(theta) * dt
    theta += v / L * jnp.tan(delta) * dt
    v += a * dt

    return jnp.array([x, y, theta, v])


def lax_wrapper_step(carry, action):
    """Wrapper of a step function."""
    state = carry[0]
    next_state = step(state, action)
    carry = (next_state,)
    return carry, next_state


def build_rollout_fn(rollout_fn_step):
    def rollout_fn(state, control):
        carry = (state,)
        _, states = jax.lax.scan(rollout_fn_step, carry, control)
        return states

    # vectorize rollout function
    func = jax.jit(jax.vmap(rollout_fn, in_axes=(None, 0)))
    return func


class Controller(ABC):
    """MPPI controller base class.

    Parameters
    ----------
    config : dict
        Controller parameters.
    """

    def __init__(self, config):
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg["seed"])
        self.rollout_fn = build_rollout_fn(lax_wrapper_step)

        self.act_dim = self.cfg["act_dim"]
        self.act_max = self.cfg["act_max"]
        self.act_min = self.cfg["act_min"]

    @abstractmethod
    def reset(self):
        """Reset the control trajectory at the start of an episode."""

    @abstractmethod
    def get_action(self, obs):
        """Get the next optimal action based on current state observation."""

    @abstractmethod
    def _sample_noise(self):
        """Get noise for constructing perturbed action sequences."""


class MPPI(Controller):
    """Model Predictive Path Integral (MPPI) controller."""

    def __init__(self, config):
        super().__init__(config)
        self.plan = None
        self.reset()

    def reset(self):
        self.plan = np.zeros((self.cfg["horizon"], self.act_dim))

    def _sample_noise(self):
        size = (self.cfg["n_samples"], self.cfg["horizon"], self.act_dim)
        return self.rng.normal(size=size) * self.cfg["noise_sigma"]

    def _compute_cost(self, state_sequences, action_sequences, target):
        # state_sequences: N x H x 4
        # action_sequences: N x H x 2
        # target: 2

        # cost is how close we are to the target during the trajectory
        distances = jnp.linalg.norm(state_sequences[:, :, :2] - target, axis=-1)
        cost = jnp.sum(distances, axis=1)

        return cost

    def get_action(self, obs):
        acts = self.plan + self._sample_noise()
        acts = np.clip(acts, self.act_min, self.act_max)

        trajectories = self.rollout_fn(obs, acts)

        target = self.cfg["current_waypoint"]
        # cost is how close we are to the target during the trajectory
        costs = self._compute_cost(trajectories, acts, target)

        exp_costs = np.exp(self.cfg["temperature"] * (np.min(costs) - costs))
        denom = np.sum(exp_costs) + 1e-10

        weighted_inputs = exp_costs[:, np.newaxis, np.newaxis] * acts
        sol = np.sum(weighted_inputs, axis=0) / denom

        self.plan = np.roll(sol, shift=-1, axis=0)
        self.plan[-1] = sol[-1]
        return sol[0]


if __name__ == "__main__":
    # test MPPI controller
    config = {
        "horizon": 120,
        "n_samples": 2048,
        "noise_sigma": 1.5,
        "temperature": 1.0,
        "act_dim": 2,
        "act_max": np.array([1.2, 1.0]),
        "act_min": np.array([-1.2, -1.0]),
        "seed": 0,
        # simulation related
        "N_SIMULATION": 1000,
        # waypoints related
        "accept_waypoint_dist": 0.1,  # distance to waypoint to accept it
        "current_waypoint": None,
    }

    # define a set of waypoints
    waypoints = jnp.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )
    waypoint_idx = 0
    config["current_waypoint"] = waypoints[waypoint_idx]

    mppi = MPPI(config)
    state = np.array([0.0, 0.0, 0.0, 0.0])
    states_history = jnp.array([state])
    target = jnp.array([10.0, 10.0])

    for _ in tqdm(range(config["N_SIMULATION"]), desc="Simulating"):
        action = mppi.get_action(state)
        state = step(state, action)
        states_history = jnp.vstack([states_history, state])

        # check if we reached the waypoint
        if (
            jnp.linalg.norm(state[:2] - config["current_waypoint"])
            < config["accept_waypoint_dist"]
        ):
            waypoint_idx += 1
            if waypoint_idx >= waypoints.shape[0]:
                waypoint_idx = 0
            config["current_waypoint"] = waypoints[waypoint_idx]
            print(f"Reached waypoint {waypoint_idx}")

    import matplotlib.pyplot as plt

    print("Simulation done.")
    plt.figure(figsize=(10, 6))
    plt.plot(states_history[:, 0], states_history[:, 1], label="Trajectory")
    plt.quiver(
        states_history[:-1, 0],
        states_history[:-1, 1],
        jnp.cos(states_history[:-1, 2]),
        jnp.sin(states_history[:-1, 2]),
        scale=20,
        width=0.003,
        color="r",
        label="Direction",
    )
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("Bicycle Model Trajectory")
    plt.legend()
    plt.grid()
    plt.show()

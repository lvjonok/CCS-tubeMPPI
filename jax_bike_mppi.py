import jax
import numpy as np
import jax.numpy as jnp
from abc import ABC, abstractmethod
from tqdm import tqdm
from track2obstacles import ConvexHull, smooth_track, generate_boundaries, add_obstacles
import matplotlib.pyplot as plt


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

    def _process_waypoints(self, state):
        # check if we reached the waypoint
        waypoint_idx = self.cfg["waypoint_idx"]
        current_waypoint = self.cfg["waypoints"][waypoint_idx]
        if (
            jnp.linalg.norm(state[:2] - current_waypoint)
            < self.cfg["accept_waypoint_dist"]
        ):
            waypoint_idx += 1
            if waypoint_idx >= self.cfg["waypoints"].shape[0]:
                waypoint_idx = 0
            print(
                f"Reached waypoint {waypoint_idx} next one is {self.cfg['waypoints'][waypoint_idx]}"
            )
            self.cfg["waypoint_idx"] = waypoint_idx

    def _compute_cost(self, state_sequences, action_sequences):
        # state_sequences: N x H x 4
        # action_sequences: N x H x 2
        # target: 2
        cost = 0

        current_target_idx = self.cfg["waypoint_idx"]
        closest_target = self.cfg["waypoints"][current_target_idx]

        # cost is how close we are to the target during the trajectory
        distances = jnp.linalg.norm(state_sequences[:, :, :2] - closest_target, axis=-1)
        reached = distances < self.cfg["accept_waypoint_dist"]
        # discount cost if we reached the target
        discounted = jnp.where(reached, 0.0, distances)
        cost += jnp.sum(discounted, axis=1)
        # cost += jnp.sum(distances, axis=1)

        # we want to have direction aligned with the target, more important at the end
        angles = jnp.arctan2(
            closest_target[1] - state_sequences[:, :, 1],
            closest_target[0] - state_sequences[:, :, 0],
        )
        angle_weights = jnp.geomspace(0.1, 1.0, state_sequences.shape[1])
        angle_cost = jnp.sum(
            angle_weights * jnp.abs(angles - state_sequences[:, :, 2]), axis=1
        )
        cost += angle_cost

        # we want to keep average speed positive and close to 3.0
        speed = jnp.mean(state_sequences[:, :, 3], axis=1)
        speed_cost = (speed - self.cfg["target_velocity"]) ** 2
        cost += speed_cost

        # we want to avoid obstacles
        # hard constraint on obstacles
        obs_distances = jnp.linalg.norm(
            state_sequences[:, :, :2][:, :, None, :] - self.cfg["obstacles"][None],
            axis=-1,
        )
        # if we are inside an obstacle, cost is high
        collision = jnp.any(obs_distances < self.cfg["obstacle_radius"], axis=-1)
        collision_cost = jnp.sum(collision, axis=1) * 1e4
        cost += collision_cost

        return cost

    def get_action(self, obs):
        acts = self.plan + self._sample_noise()
        acts = np.clip(acts, self.act_min, self.act_max)

        trajectories = self.rollout_fn(obs, acts)

        costs = self._compute_cost(trajectories, acts)

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
        "horizon": 200,
        "n_samples": 512,
        "noise_sigma": 3.0,
        "temperature": 1.0,
        "act_dim": 2,
        "act_max": np.array([1.4, 2.0]),
        "act_min": np.array([-1.4, -2.0]),
        "seed": 0,
        # simulation related
        "N_SIMULATION": 6000,
        # waypoints related
        # distance to waypoint to accept it
        "accept_waypoint_dist": 0.2,
        "target_velocity": 4.0,
        "waypoints": None,
        "waypoint_idx": 0,
        # obstacles parameters
        "obstacle_radius": 0.15,
        "obstacles": None,
    }

    # create random racing track
    TRACK_WIDTH = 0.60
    points = jnp.array(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 2.0],
            [5.0, 3.0],
            [6.0, 3.0],
            [7.0, 4.0],
            [6.0, 5.0],
            [5.0, 5.0],
            [4.0, 6.0],
            [3.0, 6.0],
            [2.0, 5.0],
            [1.0, 5.0],
            [1.0, 4.0],
        ]
    )

    # Step 2: Compute the convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # Step 3: Smooth the track
    smooth_points = smooth_track(hull_points)
    # Add obstacles along boundaries
    left_boundary, right_boundary = generate_boundaries(smooth_points, TRACK_WIDTH)
    config["obstacles"] = jnp.vstack([left_boundary, right_boundary])

    # define a set of waypoints
    waypoints = smooth_points[::15]
    config["waypoints"] = waypoints

    mppi = MPPI(config)
    # find direction from first to second waypoint
    dir0 = waypoints[1] - waypoints[0]
    theta0 = jnp.arctan2(dir0[1], dir0[0])
    state = np.array([*smooth_points[0], theta0, 0.0])
    states_history = jnp.array([state])

    for _ in tqdm(range(config["N_SIMULATION"]), desc="Simulating"):
        action = mppi.get_action(state)
        state = step(state, action)
        states_history = jnp.vstack([states_history, state])

        mppi._process_waypoints(state)

    print("Simulation done.")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        states_history[:, 0],
        states_history[:, 1],
        label="Trajectory",
        color="blue",
    )
    ax.quiver(
        states_history[:-1, 0],
        states_history[:-1, 1],
        jnp.cos(states_history[:-1, 2]),
        jnp.sin(states_history[:-1, 2]),
        scale=20,
        width=0.003,
        color="r",
        label="Direction",
    )
    ax.plot(
        smooth_points[:, 0],
        smooth_points[:, 1],
        color="black",
        label="Central Line",
        linewidth=2,
    )

    discretization = 3  # simple const for obstacle density
    add_obstacles(
        left_boundary,
        discretization,
        mppi.cfg["obstacle_radius"],
        ax,
        color="gray",
    )
    add_obstacles(
        right_boundary,
        discretization,
        mppi.cfg["obstacle_radius"],
        ax,
        color="gray",
    )

    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title("Bicycle Model Trajectory")
    ax.legend()
    ax.grid()
    plt.show()

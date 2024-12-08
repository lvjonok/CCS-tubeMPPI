import jax
import numpy as np
import jax.numpy as jnp
from abc import ABC, abstractmethod
from tqdm import tqdm
from track2obstacles import ConvexHull, smooth_track, generate_boundaries, add_obstacles
import matplotlib.pyplot as plt
from controllers.LinCovSteer import linCovSteer
from ccs import formulate_problem, solve_problem


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


# define linearized dynamics of bicycle model
def linearized_dyn(state0, control0):
    # we linearize the dynamics around the given state and control
    # state: [x, y, theta, v]
    # control: [delta, a]

    # parameters
    L = 1.0  # length of vehicle
    dt = 0.01  # time step

    # state
    x, y, theta, v = state0
    delta, a = control0

    # linearized dynamics
    A = jnp.array(
        [
            [1.0, 0.0, -v * jnp.sin(theta) * dt, jnp.cos(theta) * dt],
            [0.0, 1.0, v * jnp.cos(theta) * dt, jnp.sin(theta) * dt],
            [0.0, 0.0, 1.0, jnp.tan(delta) / L * dt],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    B = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [v / (jnp.cos(delta) ** 2 * L) * dt, 0.0],
            [0.0, dt],
        ]
    )

    return A, B


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

        # # we want to have direction aligned with the target, more important at the end
        # angles = jnp.arctan2(
        #     closest_target[1] - state_sequences[:, :, 1],
        #     closest_target[0] - state_sequences[:, :, 0],
        # )
        # angle_weights = jnp.geomspace(0.1, 1.0, state_sequences.shape[1])
        # angle_cost = jnp.sum(
        #     angle_weights * jnp.abs(angles - state_sequences[:, :, 2]), axis=1
        # )
        # cost += angle_cost

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
        self.sol = np.sum(weighted_inputs, axis=0) / denom

        self.plan = np.roll(self.sol, shift=-1, axis=0)
        self.plan[-1] = self.sol[-1]
        return self.sol[0]

    @property
    def controls(self):
        return self.sol

    def nominal_states(self, state0, N=None):
        # using the controls we have predicted during mppi iteration
        # we perform rollout to get where we would be
        # if we followed the predicted controls
        state = state0
        states = [state]

        for i in range(N or self.cfg["horizon"]):
            state = step(state, self.sol[i])
            states.append(state)

        return jnp.array(states)


# @jax.jit
def getObsConstr(Xlist, N, obstacles, max_obstacles: int):
    """
    Function to generate obstacle avoidance constraints for covariance steering
    """
    R = 0.15
    threshold = 0.4

    nx = Xlist[0].shape[0]
    constrData = []
    help_points = []

    if max_obstacles == 0:
        return constrData, help_points

    for k in range(N + 1):
        Fpos_k = jnp.zeros((2, (N + 1) * nx))
        Fpos_k = Fpos_k.at[:, k * nx : k * nx + 2].set(jnp.eye(2))
        # help_points.append([])
        for obs_tuple in obstacles:
            # we should consider obstacle if closer than threshold
            dist = jnp.linalg.norm(Xlist[k][0:2] - obs_tuple)
            if dist > threshold:
                continue

            z0 = jnp.array([obs_tuple]).T
            mu_k_prev = Xlist[k][0:2]
            deltaz = mu_k_prev - z0
            norm = jnp.linalg.norm(deltaz, 2)
            deltaz = deltaz / norm
            xbark = z0 + deltaz * R

            # help_points[-1].append(xbark)

            a_k = deltaz
            b_k = a_k.T @ xbark

            atildek = Fpos_k.T @ a_k
            btildek = b_k
            constrData.append((atildek, btildek))

            if len(constrData) == max_obstacles:
                return constrData, help_points
    return constrData, help_points


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
        "N_SIMULATION": 600,
        # waypoints related
        # distance to waypoint to accept it
        "accept_waypoint_dist": 0.2,
        "target_velocity": 4.0,
        "waypoints": None,
        "waypoint_idx": 0,
        # obstacles parameters
        "obstacle_radius": 0.15,
        "obstacles": None,
        "obstacles_discretization": 3,
        # Constrained Covariance Steering
        "enable_ccs": False,
        "T_CS": 4,  # number of time steps for CCS to look ahead
        "Q": np.eye(4) * 1e-4,  # state cost
        "R": np.eye(2) * 1e-4,  # control cost
        "dk": np.zeros((4, 1)),  # state deviation
        "Wk": np.eye(4) * 1e-3,  # state deviation covariance
        # when after CCS maximum eigenvalue is above this threshold
        # the nominal state is set equal to the real state
        # and the covariance is set to 0
        "sigma_threshold": 1.0,
        "max_obstacles": 0,
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
    # take only each 5th point to reduce obstacle density
    left_boundary = left_boundary[:: config["obstacles_discretization"]]
    right_boundary = right_boundary[:: config["obstacles_discretization"]]
    config["obstacles"] = jnp.vstack([left_boundary, right_boundary])

    # define a set of waypoints
    waypoints = smooth_points[::15]
    config["waypoints"] = waypoints

    mppi = MPPI(config)
    # find direction from first to second waypoint
    dir0 = waypoints[1] - waypoints[0]
    theta0 = jnp.arctan2(dir0[1], dir0[0])
    state = np.array([*smooth_points[0], theta0, 0.0])

    # now we are ready to run the loop for the algorithm
    state_nominal = state.copy()  # we keep track of the nominal state
    SigmaK = np.eye(4) * 0  # initial covariance is 0
    # initialize the controller for the nominal state
    # and get initial nominal control sequence
    mppi.get_action(state_nominal)
    mppi._process_waypoints(state_nominal)
    controls, controls_nominal = mppi.controls.copy(), mppi.controls.copy()

    # formulate cvxpy problem for CCS
    ccs_prob = formulate_problem(
        nx=4,
        nu=2,
        N=config["T_CS"],
        N_obs=config["max_obstacles"],
    )

    Wk = np.eye(4) * 1e-3
    Wk[:2, :2] = 0.0

    states_history = jnp.array([state])
    states_nominal_history = jnp.array([state_nominal])

    for _ in tqdm(range(config["N_SIMULATION"]), desc="Simulating"):
        # run mppi for the nominal state
        action = mppi.get_action(state_nominal)

        # part about CCS
        if config["enable_ccs"]:
            # get nominal trajectory and controls
            nominal_controls = mppi.controls
            nominal_states = mppi.nominal_states(state_nominal, N=config["T_CS"])
            # generate a list of linearized dynamics for the nominal trajectory
            Alist = []
            Blist = []
            for i in range(config["T_CS"]):
                A, B = linearized_dyn(nominal_states[i], nominal_controls[i])
                Alist.append(A)
                Blist.append(B)

            d = np.zeros((4, 1))
            W = np.eye(4) * 0

            dlist = [d] * config["T_CS"]
            Wlist = [W] * config["T_CS"]

            # compute obstacles constraints for nominal trajectory
            constrData, _ = getObsConstr(
                Xlist=nominal_states,
                N=config["T_CS"],
                obstacles=config["obstacles"],
                max_obstacles=config["max_obstacles"],
            )
            constrData = [(np.array(a), np.array(b)) for a, b in constrData]

            # have to define Qlist and Rlist
            Q = np.eye(4) * 1
            R = np.eye(2) * 1e-2
            Qlist = [Q] * (config["T_CS"] + 1)
            Rlist = [R] * config["T_CS"]

            uff_, L_, K_, prob_status = solve_problem(
                *ccs_prob,
                Alist=np.array(Alist),
                Blist=np.array(Blist),
                dlist=dlist,
                Wlist=Wlist,
                mu_0=np.array(state_nominal).reshape(-1, 1),
                Sigma_0=SigmaK,
                Qlist=Qlist,
                Rlist=Rlist,
                Xref=np.expand_dims(nominal_states[: config["T_CS"] + 1], axis=-1),
                Uref=np.expand_dims(nominal_controls[: config["T_CS"]], axis=-1),
                ObsAvoidConst=constrData,
            )
            if prob_status != "optimal":
                print("Optimization failed", prob_status)
                break

            ubark, Kfbk = uff_[0:2, :], L_[0:2, :]
            uk = ubark + Kfbk @ np.array(state - state_nominal).reshape(-1, 1)

            # fix the dimensions
            ubark = ubark.T.flatten()
            uk = uk.T.flatten()
            Ak, Bk = linearized_dyn(state_nominal, ubark)
            Sigmakp1 = ((Ak + Bk @ Kfbk) @ SigmaK @ (Ak + Bk @ Kfbk).T) + Wk
            SigmaK = Sigmakp1
        else:
            # if no CCS we just use the nominal control
            uk = action
            ubark = action

        wk = np.random.multivariate_normal(np.zeros(4), Wk, (1,)).flatten() * 0

        # Stepping simulation forward
        state = step(state, uk) + wk
        state_nominal = step(state_nominal, ubark)

        # update state and nominal state
        states_history = jnp.vstack([states_history, state])
        states_nominal_history = jnp.vstack([states_nominal_history, state_nominal])

        mppi._process_waypoints(state)

        # also CCS part
        # check if we need to reset the nominal state
        # if the maximum eigenvalue of the covariance is above the threshold
        if np.max(np.linalg.eigvals(SigmaK)) > config["sigma_threshold"]:
            state_nominal = state.copy()
            SigmaK = np.eye(4) * 0

    print("Simulation done.")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        smooth_points[:, 0],
        smooth_points[:, 1],
        color="black",
        label="Central Line",
        linewidth=2,
        linestyle="--",
        alpha=0.5,
    )
    ax.plot(
        states_history[:, 0],
        states_history[:, 1],
        label="Trajectory",
        color="blue",
    )

    add_obstacles(
        left_boundary,
        config["obstacles_discretization"],
        mppi.cfg["obstacle_radius"],
        ax,
        color="gray",
    )
    add_obstacles(
        right_boundary,
        config["obstacles_discretization"],
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
    print("Done.")

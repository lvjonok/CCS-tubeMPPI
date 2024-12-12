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

    def pure_pursuit_control(
        self, state, waypoints, current_target_idx, k=0.1, L=1.0, Lfc=2.0
    ):
        rear_x = state[:, :, 0] - ((L / 2) * jnp.cos(state[:, :, 2]))
        rear_y = state[:, :, 1] - ((L / 2) * jnp.sin(state[:, :, 2]))
        alpha = (
            jnp.arctan2(
                waypoints[current_target_idx][1] - rear_y,
                waypoints[current_target_idx][0] - rear_x,
            )
            - state[:, :, 2]
        )
        delta = jnp.arctan2(2 * L * jnp.sin(alpha) / (k * state[:, :, 3] + Lfc), 1.0)
        return delta

    def stanley_control(self, state, waypoints, current_target_idx, k=0.5, L=1.0):
        def calc_target_index(state, cx, cy):
            """
            Compute index in the trajectory list of the target.

            :param state: (State object)
            :param cx: [float]
            :param cy: [float]
            :return: (int, float)
            """
            # Calc front axle position
            fx = state[:, :, 0] + L * jnp.cos(state[:, :, 2])
            fy = state[:, :, 1] + L * jnp.sin(state[:, :, 2])

            # Search nearest point index
            dx = jnp.array([fx - icx for icx in cx])
            dy = jnp.array([fy - icy for icy in cy])
            d = jnp.hypot(dx, dy)
            target_idx = jnp.argmin(d, axis=0)

            # Project RMS error onto front axle vector
            front_axle_vec = jnp.stack(
                [
                    -jnp.cos(state[:, :, 2] + jnp.pi / 2),
                    -jnp.sin(state[:, :, 2] + jnp.pi / 2),
                ]
            )

            error_front_axles = []

            for i in range(state.shape[0]):
                for j in range(state.shape[1]):
                    error_front_axles.append(
                        (
                            dx[target_idx[i, j]] * front_axle_vec[0]
                            + dy[target_idx[i, j]] * front_axle_vec[1]
                        ).mean()
                    )

            return target_idx, jnp.array(error_front_axles).reshape(
                state.shape[0], state.shape[1]
            )

        def normalize_angle(x, zero_2_2pi=False, degree=False):
            if isinstance(x, float):
                is_float = True
            else:
                is_float = False

            x = jnp.asarray(x).flatten()
            if degree:
                x = jnp.deg2rad(x)

            if zero_2_2pi:
                mod_angle = x % (2 * jnp.pi)
            else:
                mod_angle = (x + jnp.pi) % (2 * jnp.pi) - jnp.pi

            if degree:
                mod_angle = jnp.rad2deg(mod_angle)

            if is_float:
                return mod_angle.item()
            else:
                return mod_angle

        _, error_front_axle = calc_target_index(state, waypoints[:, 0], waypoints[:, 1])

        # theta_e corrects the heading error
        theta_e = normalize_angle(
            jnp.arctan2(
                waypoints[current_target_idx][1], waypoints[current_target_idx][0]
            )
            - state[:, :, 2]
        ).reshape(state.shape[:2])
        # theta_d corrects the cross track error
        theta_d = np.arctan2(k * error_front_axle, state[:, :, 3])
        # Steering control
        delta = theta_e + theta_d

        return delta

    def _compute_cost(
        self,
        state_sequences,
        action_sequences,
        weight_angle=1e2,
        angle_control="none",
    ):
        # state_sequences: N x H x 4
        # action_sequences: N x H x 2
        # target: 2
        cost = 0

        current_target_idx = self.cfg["waypoint_idx"]
        closest_target = self.cfg["waypoints"][current_target_idx]

        # cost is how close we are to the target during the trajectory
        distances = jnp.linalg.norm(state_sequences[:, :, :2] - closest_target, axis=-1)
        # add cost only for the first point
        reached = distances < self.cfg["accept_waypoint_dist"]
        # discount cost if we reached the target
        discounted = jnp.where(reached, 0.0, distances)
        cost += jnp.sum(discounted, axis=1)

        # # then we add the remaining distances, but geometrically decreasing
        # not_reached = jnp.where(reached, 0.0, 1.0)
        # geom = jnp.geomspace(1.0, 0.1, state_sequences.shape[1])
        # cost += jnp.sum(not_reached * geom * distances, axis=1)

        if angle_control == "stanley":
            cost += (
                0.5
                * weight_angle
                * self.stanley_control(
                    state_sequences, self.cfg["waypoints"], current_target_idx
                )[0].mean(axis=1)
            )
        elif angle_control == "pure_pursuit":
            delta = self.pure_pursuit_control(
                state_sequences, self.cfg["waypoints"], current_target_idx
            )

            cost += (
                0.5
                * weight_angle
                * jnp.linalg.norm(delta - action_sequences[:, :, 0], axis=1)
            )

            # cost += (
            #     0.5
            #     * weight_angle
            #     * self.pure_pursuit_control(
            #         state_sequences, self.cfg["waypoints"], current_target_idx
            #     ).mean(axis=1)
            # )
        else:
            pass
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
        if self.cfg["enable_obstacles"]:
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

        # print(self.cfg["angle_control"])
        costs = self._compute_cost(
            trajectories,
            acts,
            angle_control=self.cfg["angle_control"],
        )

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
    # parse the desired name of the sim data file
    import argparse

    parser = argparse.ArgumentParser(
        description="Run MPPI controller on a racing track."
    )
    parser.add_argument(
        "--sim_data",
        type=str,
        default="sim_data.npz",
        help="Name of the simulation data file.",
    )

    args = parser.parse_args()

    # predefine some racing tracks
    racing_tracks = {
        "track1": jnp.array(
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
        ),
        "track2": jnp.array(
            [
                [0, 0],
                [5.00, 0],
                [5.00, 0.5],
                [5.00, 1.5],
                [5.00, 5.00],
                [10.00, 10.00],
                [9.00, 10.00],
                [5.00, 10.00],
                [15.00, 0],
            ]
        ),
    }

    # test MPPI controller
    config = {
        # track config
        "track": "track2",
        # MPPI config
        "horizon": 200,
        "n_samples": 128,
        "noise_sigma": 3.0,
        "temperature": 1.0,
        "act_dim": 2,
        "act_max": np.array([1.4, 2.0]),
        "act_min": np.array([-1.4, -2.0]),
        "seed": 0,
        # simulation related
        "N_SIMULATION": 5000,
        # waypoints related
        # distance to waypoint to accept it
        "accept_waypoint_dist": 0.3,
        "target_velocity": 4.0,
        "waypoints": None,
        "waypoint_idx": 0,
        "waypoints_discretization": 25,
        # obstacles parameters
        "enable_obstacles": True,
        "obstacle_radius": 0.15,
        "obstacles": None,
        "obstacles_discretization": 2,
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
        # Choose between 'stanley' and 'pure_pursuit'
        "angle_control": "none",  # 'pure_pursuit'
        "sim_data": args.sim_data,
    }

    # create random racing track
    TRACK_WIDTH = 0.60
    points = racing_tracks[config["track"]]

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
    if config["enable_obstacles"]:
        config["obstacles"] = jnp.vstack([left_boundary, right_boundary])

    # define a set of waypoints
    waypoints = smooth_points[:: config["waypoints_discretization"]]
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
    control_history = jnp.array([mppi.get_action(state_nominal)])
    states_nominal_history = jnp.array([state_nominal])

    for _ in tqdm(range(config["N_SIMULATION"]), desc="Simulating"):
        # run mppi for the nominal state
        action = mppi.get_action(state_nominal)
        control_history = jnp.vstack([control_history, action])

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

    # save the trajectory and all the data
    sim_data = {
        "config": mppi.cfg,
        "states_history": states_history,
        "control_history": control_history,
    }
    np.savez(config["sim_data"], **sim_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    # equal aspect ratio
    ax.set_aspect("equal")
    # add waypoints as small circles
    ax.scatter(
        waypoints[:, 0],
        waypoints[:, 1],
        color="red",
        label="Waypoints",
        s=100,
        zorder=10,
    )
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

    if config["enable_obstacles"]:
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

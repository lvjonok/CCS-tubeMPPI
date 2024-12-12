# CSMPPI 2d quadrotor test script

import numpy as np
from costFunctions.costfun import QuadHardCost, QuadSoftCost, QuadSoftCost2
from costFunctions.costfun import QuadObsCost, QuadPosCost

from sysDynamics.sysdyn import car_dynamics

from controllers.MPPI import MPPI_pathos

from Plotting.plotdata import plot_quad

from matplotlib import pyplot as plt

from tqdm import tqdm
import argparse
import os

from track2obstacles import ConvexHull, smooth_track, generate_boundaries


def main():
    parser = argparse.ArgumentParser(
        "Covariance Steering MPPI for 2d " + "quadrotor obstacle avoidance "
    )
    parser.add_argument("-mu", help="Mu parameter for MPPI", default=1.0, type=float)
    parser.add_argument(
        "-nu",
        default=1.0,
        type=float,
        help="Nu parameter for Sampling "
        + "with higher variance default=1., pick >=1.",
    )
    parser.add_argument("-K", help="MPPI sample size parameter", default=500, type=int)
    parser.add_argument("-T", help="MPPI horizon parameter", default=10, type=int)
    parser.add_argument("-Tsim", help="Simulation Time steps", default=200, type=int)
    parser.add_argument(
        "-lambda",
        dest="LAMBDA",
        default=0.1,
        type=float,
        help="Cost Function Parameter lambda default=0.1",
    )
    parser.add_argument(
        "-dt", type=float, default=0.05, help="Discrete time step. Default dt=0.05"
    )
    parser.add_argument(
        "-Rexit",
        type=float,
        default=20.0,
        help="Simulation exit limits if abs(px) or abs(py) "
        + ">= Rexit, then the simulation is terminated. "
        + "Default=20.",
    )
    parser.add_argument(
        "-seed", type=int, default=100, help="Random Number Generator Seed"
    )
    parser.add_argument(
        "-no-noise",
        default=False,
        action="store_true",
        dest="nonoise",
        help="Flag to simulate without noise on the input",
    )
    parser.add_argument(
        "-add-noise",
        type=float,
        default=0.0,
        dest="addnoise",
        help="additional noise to the system",
    )
    parser.add_argument(
        "-paramfile",
        default="./quad_params/quad_params1.txt",
        help="parameters file directory for simulations",
    )
    parser.add_argument(
        "-filename", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument(
        "-qmult",
        type=float,
        default=1.0,
        help="Multiplier of state cost function default",
    )
    parser.add_argument(
        "-des-pos",
        dest="des_pos_str",
        type=str,
        help="string of desired position given as (px, py)" + " default : (5., 5.)",
        default="(5., 5.)",
    )
    parser.add_argument(
        "-obs-file",
        dest="obs_file",
        help="obstacles file. " + "default = './quad_params/obs1.npy'",
        default="./quad_params/obs1.npy",
    )
    parser.add_argument(
        "-cost",
        type=str,
        default="hard",
        choices=["sep", "hard", "soft", "soft2"],
        help="Cost Type. Default:sep, " + "options: sep, hard, soft, soft2",
    )
    args = parser.parse_args()

    mu = args.mu
    NU_MPPI = args.nu
    K = args.K
    T = args.T
    iteration = args.Tsim
    dt = args.dt
    lambda_ = args.LAMBDA
    seed = args.seed
    ADD_NOISE = args.addnoise
    Q_MULT = args.qmult
    des_pos_str = args.des_pos_str
    DES_POS_LIST = des_pos_str.replace("(", "").replace(")", "").split(",")
    DES_POS = tuple([float(x) for x in DES_POS_LIST])
    Rexit = args.Rexit
    OBS_FILE = args.obs_file
    COST_TYPE = args.cost

    np.random.seed(seed)

    FILENAME = args.filename

    PARAMFILE = args.paramfile
    if os.path.exists(PARAMFILE):
        with open(PARAMFILE) as f:
            filelist = f.readlines()

        for line in filelist:
            if "Natural System Noise Parameter" in line:
                mu = float(line.split(":")[1])
            elif "Control Sampling Covariance Parameter" in line:
                NU_MPPI = float(line.split(":")[1])
            elif "Number of Samples" in line:
                K = int(line.split(":")[1])
            elif "MPC Horizon" in line:
                T = int(line.split(":")[1])
            elif "Number of Simulation Timesteps" in line:
                iteration = int(line.split(":")[1])
            elif "Discretization time-step" in line:
                dt = float(line.split(":")[1])
            elif "Control Cost Parameter" in line:
                lambda_ = float(line.split(":")[1])
            elif "Random Number Generator" in line:
                seed = int(line.split(":")[1])
            elif "Q Multiplier" in line:
                Q_MULT = float(line.split(":")[1])
            elif "Additional Noise" in line:
                ADD_NOISE = float(line.split(":")[1])
            elif "Desired Position" in line:
                des_pos_string = line.split(":")[1]
                des_list = des_pos_string.split("(")[1].replace(")", "").split(",")
                DES_POS = tuple([float(x) for x in des_list])
            elif "Obstacle File" in line:
                OBS_FILE = line.split("'")[1]
            elif "Cost Type" in line:
                COST_TYPE = line.split(":")[1].replace(" ", "").replace("\n", "")

    x0 = np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])

    Sigma = mu * np.eye(2)
    Sigmainv = np.linalg.inv(Sigma)
    Ubar = np.ones((2, T))

    F = lambda x, u: car_dynamics(x, u)

    obs_list = np.load(OBS_FILE, allow_pickle=True)
    print(obs_list)

    # WE GENERATE CUSTOM TRACK
    TRACK_WIDTH = 0.60
    points = np.array(
        [
            [1, 1],
            [2, 1],
            [3, 2],
            [4, 2],
            [5, 3],
            [6, 3],
            [7, 4],
            [6, 5],
            [5, 5],
            [4, 6],
            [3, 6],
            [2, 5],
            [1, 5],
            [1, 4],
        ]
    )

    # Step 2: Compute the convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # Step 3: Smooth the track
    smooth_points = smooth_track(hull_points)

    # Step 4: Generate track boundaries
    left_boundary, right_boundary = generate_boundaries(smooth_points, TRACK_WIDTH)
    circle_radius = 0.25

    obs_list = []
    for i in range(len(left_boundary)):
        obs_list.append([(left_boundary[i, 0], left_boundary[i, 1]), circle_radius])

    for i in range(len(right_boundary)):
        obs_list.append([(right_boundary[i, 0], right_boundary[i, 1]), circle_radius])

    x0 = np.array([[1], [1], [0.0], [0.0], [0.0]])
    DES_POS = (1.3, 1)
    # exit(0)

    print(COST_TYPE)
    # if COST_TYPE == "sep":
    #     # C = lambda x: Q_MULT * QuadObsCost(x, dt, obstacles=obs_list)
    #     # Phi = lambda x: Q_MULT * T * QuadPosCost(x, dt, pdes=DES_POS)
    if COST_TYPE == "hard":
        C = lambda x: Q_MULT * QuadHardCost(x, dt, pdes=DES_POS, obstacles=obs_list)
        Phi = lambda x: 0.0
    # elif COST_TYPE == "soft":
    #     C = lambda x: Q_MULT * QuadSoftCost(x, dt, pdes=DES_POS, obstacles=obs_list)
    #     Phi = lambda x: 0.0
    # elif COST_TYPE == "soft2":
    #     C = lambda x: Q_MULT * QuadSoftCost2(x, dt, pdes=DES_POS, obstacles=obs_list)
    #     Phi = lambda x: 0.0
    else:
        print("Undefined Cost Function!!")
        exit()

    Wk = np.eye(5) * dt
    Wk[0:2, 0:2] = np.zeros((2, 2))
    Wk = Wk * ADD_NOISE

    # Qk, Rk = 100*np.eye(nx), 0.001*np.eye(nu)
    # Qk[2:,2:] = 0.1*np.eye(2)
    # Qfinal = Qk
    #
    # Alist, Blist, dlist, Wlist, Qlist, Rlist = [], [], [], [], [], []
    #
    # Alist, Blist = [Ak for k in range(T)], [Bk for k in range(T)]
    # dlist, Wlist = [dk for k in range(T)], [Wk for k in range(T)]
    # Qlist, Rlist = [Qk for k in range(T+1)], [Rk for k in range(T)]
    #
    # Tau_threshold = 0.

    Xreal = []
    Ureal = []
    Xreal.append(x0)
    xk = x0
    xk_nom = xk
    total_cost = 0.0
    Unom, U = Ubar, Ubar
    for i in tqdm(range(iteration), disable=False):
        X, U, Sreal = MPPI_pathos(
            xk,
            F,
            K,
            T,
            Sigma,
            Phi,
            C,
            lambda_,
            U,
            Nu_MPPI=NU_MPPI,
            dt=dt,
            progbar=False,
        )

        eps = np.random.multivariate_normal(np.zeros(2), np.eye(2), (1,)).T * mu

        wk = np.random.multivariate_normal(np.zeros(5), Wk, (1,)).T

        uk = U[:, 0:1]
        xkp1 = xk + F(xk, uk + eps) * dt + wk
        # normalize theta
        # xkp1[2] = (xkp1[2] + np.pi) % (2 * np.pi) - np.pi

        Xreal.append(xkp1)
        Ureal.append(uk)

        xk = xkp1

        Udummy = np.zeros(U.shape)
        Udummy[:, 0:-1] = U[:, 1:]
        Udummy[:, -1:] = U[:, -2:-1]
        U = Udummy

        # if velocity is negative, then we just temporarily stop
        if xk[3] < 0:
            print("violate positive velocity")
            break

        Rkp1 = np.linalg.norm(xkp1[0:2], 2)
        total_cost += (C(xk) + (lambda_ / 2.0) * (uk.T @ Sigmainv @ uk)) * dt
        if np.abs(xkp1[0]) >= Rexit or np.abs(xkp1[1]) >= Rexit:
            print("Major Violation of Safety, Simulation Ended prematurely")
            break

    X = np.block([Xreal])
    Xpos = X[0:2, :]
    Xvel = X[2:, :]
    Vvst = np.sqrt(np.sum(np.square(Xvel), 0))
    Vmean = np.mean(Vvst)
    U = np.block([Ureal])
    Uxvst, Uyvst = U[0:1, :].squeeze(), U[1:2, :].squeeze()

    figtraj, axtraj = plot_quad(X, obs_list, DES_POS)

    fig2, ax2 = plt.subplots()
    ax2.plot(Vvst)
    ax2.title.set_text("V vs t")

    fig3, (ax3, ax4) = plt.subplots(2)
    ax3.plot(Uxvst)
    ax3.title.set_text("$u_{x}$ vs t")
    ax4.plot(Uyvst)
    ax4.title.set_text("$u_{y}$ vs t")

    paramslist = []
    # paramslist.append('Smooth Cost with Wpos:{} and Wvel:{}'.format(Wpos, Wvel) if SOFTCOST else 'Sparse Cost')
    paramslist.append("Natural System Noise Parameter, mu : {}".format(mu))
    paramslist.append("Control Sampling Covariance Parameter, nu : {}".format(NU_MPPI))
    paramslist.append("Number of Samples, K : {}".format(K))
    paramslist.append("MPC Horizon, T : {}".format(T))
    paramslist.append(
        "Number of Simulation Timesteps, iteration : {}".format(iteration)
    )
    paramslist.append("Discretization time-step, dt : {}".format(dt))
    paramslist.append("Control Cost Parameter, Lambda : {}".format(lambda_))
    paramslist.append("Random Number Generator, seed : {}".format(seed))
    # paramslist.append('Desired Speed : {}'.format(V_DES))
    paramslist.append("Q Multiplier : {}".format(Q_MULT))
    paramslist.append("Cost Type : {}".format(COST_TYPE))
    paramslist.append("Additional Noise Parameter, W : {}".format(ADD_NOISE))
    paramslist.append("Desired Position : {}".format(DES_POS))
    paramslist.append("-------RESULTS-------")
    paramslist.append("Total Cost : {:.2f}".format(float(total_cost)))
    paramslist.append(
        "Average Cost : {:.2f}".format(float(total_cost / (iteration * dt)))
    )
    # paramslist.append('Average Speed : {:.2f}'.format(Vmean))

    if FILENAME is None:
        print("\n".join(paramslist))
        plt.show()
    elif type(FILENAME) is str:
        if not os.path.exists(FILENAME):
            os.system("mkdir {}".format(FILENAME))
        np.save(FILENAME + "/X.npy", X)
        np.save(FILENAME + "/obs_list.npy", obs_list)
        figtraj.savefig(FILENAME + "/fig_traj.pdf")
        fig2.savefig(FILENAME + "/fig_v.pdf")
        fig3.savefig(FILENAME + "/fig_u.pdf")

        with open(FILENAME + "/params.txt", "w+") as f:
            f.write("\n".join(paramslist) + "\n")

    else:
        pass

    pass


if __name__ == "__main__":
    main()

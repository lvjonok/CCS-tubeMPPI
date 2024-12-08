# Plot Circle

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["figure.autolayout"] = True


def plot_circle(X, R=2.0, h=0.125, help_points=[], X2=[]):
    Rin = R - h
    Rout = R + h
    thetadummy = np.linspace(0, 2 * np.pi, 100)
    Rindata = Rin * np.array([np.cos(thetadummy), np.sin(thetadummy)])
    Routdata = Rout * np.array([np.cos(thetadummy), np.sin(thetadummy)])

    fig, ax = plt.subplots()
    # import pdb; pdb.set_trace()
    ax.plot(Rindata[0], Rindata[1], linewidth=2, color="k")
    ax.plot(Routdata[0], Routdata[1], linewidth=2, color="k")

    px, py = X[0, :], X[1, :]
    ax.plot(px, py, linewidth=1, color="r", marker=" ")
    # draw arrows with theta
    for ii in range(len(px)):
        ax.arrow(
            px[ii],
            py[ii],
            0.5 * np.cos(X[2, ii]),
            0.5 * np.sin(X[2, ii]),
            head_width=0.1,
            head_length=0.1,
            fc="k",
            ec="k",
            alpha=1 - ii / len(px),
        )

    if help_points:
        # h1 = []
        # for ii in range(len(help_points)):
        # 	for jj in range(len(help_points[ii])):
        # 		h1.append(help_points[ii][jj])
        help_arr = np.array(help_points).squeeze().T
        x_coord = help_arr[0, :]
        y_coord = help_arr[1, :]
        ax.plot(x_coord, y_coord, "bo", markersize=5.0)
    if len(X2) > 0:
        px2, py2 = X2[0, :], X2[1, :]
        ax.plot(px2, py2, linewidth=2, color="g", linestyle="--", marker="*")

    return fig, ax


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_circle(
    X, R=2.0, h=0.125, help_points=[], X2=[], interval=100, save_path=None
):
    Rin = R - h
    Rout = R + h
    thetadummy = np.linspace(0, 2 * np.pi, 100)
    Rindata = Rin * np.array([np.cos(thetadummy), np.sin(thetadummy)])
    Routdata = Rout * np.array([np.cos(thetadummy), np.sin(thetadummy)])

    fig, ax = plt.subplots()
    ax.set_aspect("equal", "box")
    ax.set_xlim(-Rout - 1, Rout + 1)
    ax.set_ylim(-Rout - 1, Rout + 1)

    # Plot static inner and outer circles
    ax.plot(Rindata[0], Rindata[1], linewidth=2, color="k")
    ax.plot(Routdata[0], Routdata[1], linewidth=2, color="k")

    # Dynamic elements: trajectory points and current arrow
    trajectory_segments = []
    for _ in range(X.shape[1] - 1):
        (line,) = ax.plot([], [], "r-", alpha=0)  # Create individual segments
        trajectory_segments.append(line)

    current_arrow = None  # Initialize placeholder for the current arrow

    # Help points (static)
    if len(help_points) > 0:
        help_arr = np.array(help_points).squeeze().T
        (help_plot,) = ax.plot(help_arr[0, :], help_arr[1, :], "bo", markersize=5.0)
    else:
        help_plot = None

    # Static reference trajectory (X2)
    if len(X2) > 0:
        (X2_plot,) = ax.plot(X2[0, :], X2[1, :], "g--", marker="*", linewidth=2)
    else:
        X2_plot = None

    def init():
        for seg in trajectory_segments:
            seg.set_data([], [])
        return (
            trajectory_segments
            + ([] if not help_plot else [help_plot])
            + ([] if not X2_plot else [X2_plot])
        )

    def update(frame):
        nonlocal current_arrow  # Ensure we can modify the arrow object

        # Update trajectory segments with fading alpha
        for i in range(frame):
            trajectory_segments[i].set_data(
                [X[0, i], X[0, i + 1]], [X[1, i], X[1, i + 1]]
            )
            trajectory_segments[i].set_alpha((i + 1) / frame)  # Gradual fading

        # Remove the previous arrow if it exists
        if current_arrow is not None:
            current_arrow.remove()

        # Draw the new arrow
        current_arrow = ax.arrow(
            X[0, frame],
            X[1, frame],
            0.5 * np.cos(X[2, frame]),
            0.5 * np.sin(X[2, frame]),
            head_width=0.1,
            head_length=0.1,
            fc="k",
            ec="k",
        )
        return trajectory_segments + [current_arrow]

    anim = FuncAnimation(
        fig, update, frames=X.shape[1], init_func=init, blit=False, interval=interval
    )

    if save_path:
        try:
            writer = PillowWriter(fps=1000 // interval)
            anim.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Error saving animation: {e}")

    plt.show()
    return anim


def plot_quad(X, obs_list, pdes, fit_traj=False):
    thetadummy = np.linspace(0, 2 * np.pi, 100)

    fig, ax = plt.subplots()

    # Create the list of obstacle patches:
    patch_obs_list = []
    # import pdb; pdb.set_trace()
    for obs_tuple in obs_list:
        obsX, obsY, obsR = obs_tuple[0][0], obs_tuple[0][1], obs_tuple[1]
        ax.add_artist(
            mpatches.Circle((obsX, obsY), obsR, fill=True, alpha=0.8, color="k")
        )
        # patch_obs_list.append(
        # 	mpatches.Circle((obsX, obsY), obsR, fill=True, alpha=0.5, color='k')
        # 		)
    # collection_obs = PatchCollection(patch_obs_list)

    px, py = X[0, :], X[1, :]
    ax.plot(px, py, linewidth=2, color="r")
    # ax.plot(px, py, linewidth=2, color='r', marker='+')
    ax.plot(pdes[0], pdes[1], "g+", markersize=4)

    xlim_right = 10.5
    xlim_left = -0.5
    ylim_up = 10.5
    ylim_down = -0.5

    if fit_traj:
        maxpx, minpx, maxpy, minpy = max(px), min(px), max(py), min(py)

        xlim_right = max(xlim_right, maxpx)
        xlim_left = min(xlim_left, minpx)
        ylim_up = max(ylim_up, maxpy)
        ylim_down = min(ylim_down, minpy)

    ax.set_xlim([xlim_left, xlim_right])
    ax.set_ylim([ylim_down, ylim_up])

    return fig, ax


if __name__ == "__main__":
    pass

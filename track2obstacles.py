# inspired by: https://github.com/juangallostra/procedural-tracks
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy import interpolate
import random

# Constants
WIDTH, HEIGHT = 800, 600
MIN_POINTS, MAX_POINTS = 10, 20
MARGIN = 50
TRACK_WIDTH = 60
SPLINE_POINTS = 500


# Generate random points
def random_points(min_points=MIN_POINTS, max_points=MAX_POINTS, margin=MARGIN):
    point_count = random.randint(min_points, max_points)
    points = []
    while len(points) < point_count:
        x = random.randint(margin, WIDTH - margin)
        y = random.randint(margin, HEIGHT - margin)
        if all(
            np.linalg.norm(np.array([x, y]) - np.array(p)) > TRACK_WIDTH for p in points
        ):
            points.append((x, y))
    return np.array(points)


# Smooth the track using splines
def smooth_track(points):
    x = np.append(points[:, 0], points[0, 0])  # Loop back to start
    y = np.append(points[:, 1], points[0, 1])
    tck, _ = interpolate.splprep([x, y], s=0, per=True)
    xi, yi = interpolate.splev(np.linspace(0, 1, SPLINE_POINTS), tck)
    return np.array([xi, yi]).T


# Generate track boundaries
def generate_boundaries(track, track_width):
    gradients = np.gradient(track, axis=0)
    normals = np.array([[-dy, dx] for dx, dy in gradients])
    magnitudes = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= magnitudes
    left_boundary = track + normals * (track_width / 2)
    right_boundary = track - normals * (track_width / 2)
    return left_boundary, right_boundary


# Add obstacles along the boundaries
def add_obstacles(boundary, discretization, circle_radius, ax, color):
    for i in range(0, len(boundary), discretization):
        circle = plt.Circle(
            (boundary[i, 0], boundary[i, 1]), circle_radius, color=color, alpha=0.6
        )
        ax.add_artist(circle)


# Main function to create the track
def create_track(discretization=20, circle_radius=10):
    # Step 1: Generate random points
    # points = random_points()
    points = np.array(
        [
            [400, 300],
            [450, 300],
            [480, 340],
            [480, 380],
            [450, 420],
            [400, 440],
            [350, 420],
            [320, 380],
            [320, 340],
            [350, 300],
            [400, 280],
        ]
    )

    points = np.array(
        [
            [0, 0],
            [500, 0],
            [500, 5],
            [500, 15],
            [500, 500],
            [1000, 1000],
            [900, 1000],
            [500, 1000],
            [1500, 0],
        ]
    )

    # Step 2: Compute the convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # Step 3: Smooth the track
    smooth_points = smooth_track(hull_points)
    # Step 4: Generate track boundaries
    left_boundary, right_boundary = generate_boundaries(smooth_points, TRACK_WIDTH)

    # Step 5: Plot the track
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(
        smooth_points[:, 0],
        smooth_points[:, 1],
        label="Center Line",
        color="black",
        linewidth=2,
    )
    ax.plot(
        left_boundary[:, 0],
        left_boundary[:, 1],
        "--",
        label="Left Boundary",
        color="blue",
    )
    ax.plot(
        right_boundary[:, 0],
        right_boundary[:, 1],
        "--",
        label="Right Boundary",
        color="red",
    )
    ax.scatter(points[:, 0], points[:, 1], color="green", label="Random Points")
    ax.scatter(
        hull_points[:, 0],
        hull_points[:, 1],
        color="orange",
        label="Hull Points",
        zorder=5,
    )

    # Step 6: Add obstacles along boundaries
    add_obstacles(left_boundary, discretization, circle_radius, ax, color="blue")
    add_obstacles(right_boundary, discretization, circle_radius, ax, color="red")

    ax.legend()
    ax.axis("equal")
    ax.set_title("Track with Obstacles")
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    # Run the track generation
    create_track(discretization=3, circle_radius=10)

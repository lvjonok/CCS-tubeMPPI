import numpy as np
import casadi as cs


def rk4(f, xk, uk, dt):
    """
    Numerical Runge-Kutta4 method for propogating linear system dynamics:
    Inputs:
        xk -> state at time k
        uk -> control input at time k
        f  -> system dynamics : xkp1 = f(xk, uk)
        dt -> discrete time step deltat
    Output:
        xkp1 -> state at time k+1
    """
    k1 = f(xk, uk) * dt
    k2 = f(xk + k1 / 2, uk) * dt
    k3 = f(xk + k2 / 2, uk) * dt
    k4 = f(xk + k3, uk) * dt
    xkp1 = xk + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

    return xkp1


def LinearSys(xk, uk, paramdict):
    Ak = paramdict["Ak"]
    Bk = paramdict["Bk"]
    f = Ak @ xk + Bk @ uk
    return f


def integratorDyn(xk, uk):
    # x : xk[0], y : xk[1], xdot : xk[2], ydot : xk[3]
    assert xk.shape[1] == 1 and xk.shape[0] == 4
    assert uk.shape[1] == 1 and uk.shape[0] == 2

    A = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    B = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    f = A @ xk + B @ uk
    return f


def create_dynamics(L=0.1) -> cs.Function:
    state = cs.SX.sym("x", 5)
    control = cs.SX.sym("u", 2)

    heading, v, steering = state[2], state[3], state[4]  # x, y, phi, v, psi
    acc, steering_dot = control[0], control[1]  # acceleration and yaw rate

    state_dot = cs.vertcat(
        v * cs.cos(heading),
        v * cs.sin(heading),
        v * cs.tan(steering) / L,
        acc,
        steering_dot,
    )

    return cs.Function("dynamics", [state, control], [state_dot])


def create_linearized_dynamics(L=0.1) -> cs.Function:
    state = cs.SX.sym("x", 5)
    control = cs.SX.sym("u", 2)

    dyn = create_dynamics(L=L)
    state_dot = dyn(state, control)

    A = cs.jacobian(state_dot, state)
    B = cs.jacobian(state_dot, control)

    return cs.Function("linearized_dynamics", [state, control], [A, B])


__dynamics = create_dynamics()
__linearized_dynamics = create_linearized_dynamics()


def car_dynamics(xk, uk):
    # # clip input
    # uk[0] = np.clip(uk[0], -0.1, 0.4)
    # uk[1] = np.clip(uk[1], -0.5, 0.5)

    result = __dynamics(xk, uk)

    return np.array(result).reshape(-1, 1)


def lin_car_dynamics(xk, uk):
    A, B = __linearized_dynamics(xk, uk)

    return A.full(), B.full()


# def car_dynamics(xk, uk):
#     """
#     Car dynamics implementation (bicycle model)

#     Inputs:
#         xk: Current state [x, y, theta, v]
#         uk: Control input [acceleration, steering_angle]
#         paramdict: Dictionary containing parameters like wheelbase length

#     Returns:
#         f: State derivatives [dx/dt, dy/dt, dtheta/dt, dv/dt]
#     """
#     paramdict = {}

#     # Extract states
#     x, y, theta, v = xk[0, 0], xk[1, 0], xk[2, 0], xk[3, 0]

#     # Extract controls
#     a = uk[0, 0]  # acceleration
#     delta = uk[1, 0]  # steering angle

#     # clip the controls
#     a = np.clip(a, -0.1, 0.4)
#     delta = np.clip(delta, -0.5, 0.5)

#     # Get parameters
#     L = paramdict.get("wheelbase", 0.1)  # Default wheelbase length 0.1m

#     # Compute derivatives
#     dx = v * np.cos(theta)
#     dy = v * np.sin(theta)
#     dtheta = (v / L) * np.tan(delta)
#     dv = a

#     # Return state derivatives
#     f = np.array([[dx], [dy], [dtheta], [dv]])
#     return f


if __name__ == "__main__":
    pass

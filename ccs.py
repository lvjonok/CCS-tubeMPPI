# the file contains formulation of constrained covariance steering problem
# and its solution using cvxpy
from __future__ import annotations

import numpy as np
import cvxpy as cp
from scipy.stats import norm as stats_norm
# import block_diag

from scipy.linalg import block_diag
import pdb



def _phi(Alist, k2, k1):
    """
    Function to compute state transition matrix.
    k1: initial time
    k2: final time
    """
    nx = Alist[0].shape[1]
    Phi = np.eye(nx)
    for k in range(k1, k2):
        Phi = Alist[k] @ Phi
    return Phi


def formulate_problem(nx, nu, N, gamma=0, DELTA=0.01, N_obs=0):
    # define parameters we would like user to input
    # Alist = [cp.Parameter((nx, nx), name=f"A_{i}") for i in range(N)]
    # Blist = [cp.Parameter((nx, nu), name=f"B_{i}") for i in range(N)]
    # dlist = [cp.Parameter((nx, 1), name=f"d_{i}") for i in range(N)]
    # Wlist = [cp.Parameter((nx, nx), name=f"W_{i}") for i in range(N)]
    # Qlist = [cp.Parameter((nx, nx), name=f"Q_{i}") for i in range(N + 1)]
    # Rlist = [cp.Parameter((nu, nu), name=f"R_{i}") for i in range(N)]
    Xref = [cp.Parameter((nx, 1), name=f"Xref_{i}") for i in range(N + 1)]
    Uref = [cp.Parameter((nu, 1), name=f"Uref_{i}") for i in range(N)]
    mu_0 = cp.Parameter((nx, 1), name="mu_0")
    # Sigma_0 = cp.Parameter((nx, nx), name="Sigma_0")
    A_obst = [cp.Parameter(((N + 1) * nx, 2), name=f"A_obst_{i}") for i in range(N_obs)]
    B_obst = [cp.Parameter((2, 2), name=f"B_obst_{i}") for i in range(N_obs)]

    # Gu = np.zeros((nx * (N + 1), nu * N))
    # Gw = np.zeros((nx * (N + 1), nx * N))

    G0 = cp.Parameter((nx * (N + 1), nx), name="G0")
    Gw = cp.Parameter((nx * (N + 1), nx * N), name="GW")
    Gu = cp.Parameter((nx * (N + 1), nu * N), name="Gu")
    # # this is how we should define the parameter
    # for i in range(1, N + 1):
    #     G0[i * nx : (i + 1) * nx, :] = _phi(Alist, i, 0)
    #     for j in range(i):
    #         Gw[i * nx : (i + 1) * nx, j * nx : (j + 1) * nx] = _phi(Alist, i, j + 1)
    #         Gu[i * nx : (i + 1) * nx, j * nu : (j + 1) * nu] = (
    #             _phi(Alist, i, j + 1) @ Blist[j]
    #         )

    D = cp.Parameter((nx * N, 1), name="D")
    # D = np.zeros((nx * N, 1))
    # # this is how we should define the parameter
    # for i in range(N):
    #     D[i * nx : (i + 1) * nx, :] = dlist[i]

    W = cp.Parameter((N * nx, N * nx), name="W")
    # W = np.zeros((N * nx, N * nx))
    # # this is how we should define the parameter
    # for i in range(N):
    #     W[i * nx : (i + 1) * nx, i * nx : (i + 1) * nx] = Wlist[i]

    ufflist, Llist, Klist = [], [], []
    for i in range(N):
        ufflist.append(cp.Variable((nu, 1)))

    for i in range(N):
        Llist.append([])
        Llist[-1].append(cp.Variable((nu, nx)))

    for i in range(N):
        Klist.append([])
        for j in range(N):
            if j <= i - 1 and j >= (i - 1) - gamma:
                Klist[-1].append(cp.Variable((nu, nx)))
            else:
                Klist[-1].append(np.zeros((nu, nx)))

    # pdb.set_trace()
    uffvar = cp.vstack(ufflist)
    Kvar = cp.bmat(Klist)
    Lvar = cp.bmat(Llist)

    # S = Rs @ Rs.T
    # # This is how to define the parameter Rs
    # S = block_diag(Sigma_0, W)
    # u, s, vh = np.linalg.svd(S, hermitian=True)
    # Rs = u @ np.diag(np.sqrt(s))
    Rs = cp.Parameter((nx * (N + 1), nx * (N + 1)), name="Rs")

    # Cov(X) = halfcovX @ halfcovX.T
    halfcovX = cp.hstack([G0 + Gu @ Lvar, Gw + Gu @ Kvar]) @ Rs
    # Cov(Xfinal) = zetaK @ zetaK.T
    # Cov(U) = halfcovU @ halfcovU.T
    halfcovU = cp.hstack([Lvar, Kvar]) @ Rs
    # E[X] = fUbar
    fUbar = (Gu @ uffvar) + (G0 @ mu_0) + (Gw @ D)

    # u, s, vh = np.linalg.svd(W, hermitian=True)

    # u, s, vh = np.linalg.svd(Sigma_0, hermitian=True)

    # Qbig, Rbig = np.array([]), np.array([])
    # for i in range(N):
    #     if i == 0:
    #         Qbig = Qlist[i]
    #         Rbig = Rlist[i]
    #     else:
    #         Qbig = block_diag(Qbig, Qlist[i])
    #         Rbig = block_diag(Rbig, Rlist[i])

    # Qbig = block_diag(Qbig, Qlist[N])
    Qbig = cp.Parameter((nx * (N + 1), nx * (N + 1)), name="Qbig", PSD=True)
    # Qdiag = cp.Parameter(nx * (N + 1), name="Qdiag")
    # Qbig = cp.diag(Qdiag)
    Rbig = cp.Parameter((nu * N, nu * N), name="Rbig", PSD=True)
    # Rdiag = cp.Parameter(nu * N, name="Rdiag")
    # Rbig = cp.diag(Rdiag)

    # Qbig = RQbig@RQbig.T;		Rbig = RRbig@RRbig.T
    # u, s, vh = np.linalg.svd(Qbig, hermitian=True)
    # RQbig = u @ np.diag(np.sqrt(s))
    RQbig = cp.Parameter((nx * (N + 1), nx * (N + 1)), name="RQbig")

    # u, s, vh = np.linalg.svd(Rbig, hermitian=True)
    # RRbig = u @ np.diag(np.sqrt(s))
    RRbig = cp.Parameter((nu * N, nu * N), name="RRbig")

    # Turn Xref and Uref from list to array:
    Xref_array, Uref_array = cp.vstack(Xref), cp.vstack(Uref)
    DELTA_PARAM = stats_norm.ppf(1 - DELTA)
    # s = cp.Variable()

    obj_func = (
        cp.norm(RQbig.T @ halfcovX, "fro") ** 2
        + cp.quad_form(fUbar - Xref_array, Qbig)
        + cp.norm(RRbig.T @ halfcovU, "fro") ** 2
        + cp.quad_form(uffvar - Uref_array, Rbig)
    )
    obj = cp.Minimize(obj_func)
    constr = []
    # Rbig and Qbig are positive definite
    # constr.append(Rdiag >= 0)
    # constr.append(Qdiag >= 0)
    # constr.append(Rbig >> 0)
    # constr.append(Qbig >> 0)

    for constr_item_idx in range(N_obs):
        atilde = A_obst[constr_item_idx]
        btilde = B_obst[constr_item_idx]
        constr.append(
            atilde.T @ fUbar - btilde >= DELTA_PARAM * cp.norm(atilde.T @ halfcovX, 2)
        )

    prob = cp.Problem(obj, constr)

    variables = {
        "uffvar": uffvar,
        "Lvar": Lvar,
        "Kvar": Kvar,
    }

    parameters = {
        "Xref": Xref,
        "Uref": Uref,
        "mu_0": mu_0,
        "A_obst": A_obst,
        "B_obst": B_obst,
        "G0": G0,
        "Gw": Gw,
        "Gu": Gu,
        "D": D,
        "W": W,
        "Rs": Rs,
        "Qbig": Qbig,
        "Rbig": Rbig,
        "RQbig": RQbig,
        "RRbig": RRbig,
    }
    for i in range(N_obs):
        parameters[f"A_obst_{i}"] = A_obst[i]
        parameters[f"B_obst_{i}"] = B_obst[i]

    # return a problem and parameters we have to set
    return (prob, variables, parameters)


def solve_problem(
    prob: cp.Problem,
    variables: dict[str, cp.Variable],
    parameters: dict[str, cp.Parameter],
    Alist,
    Blist,
    dlist,
    Wlist,
    mu_0,
    Sigma_0,
    Qlist,
    Rlist,
    Xref,
    Uref,
    ObsAvoidConst,
    DELTA=0.01,
    gamma=0,
):
    nx, nu, N = Alist[0].shape[1], Blist[0].shape[1], len(Alist)

    # compute Gu, Gw and G0
    Gu = np.zeros((nx * (N + 1), nu * N))
    Gw = np.zeros((nx * (N + 1), nx * N))
    G0 = np.zeros((nx * (N + 1), nx))

    for i in range(1, N + 1):
        G0[i * nx : (i + 1) * nx, :] = _phi(Alist, i, 0)
        for j in range(i):
            A_ij = _phi(Alist, i, j + 1)
            Gw[i * nx : (i + 1) * nx, j * nx : (j + 1) * nx] = A_ij
            Gu[i * nx : (i + 1) * nx, j * nu : (j + 1) * nu] = A_ij @ Blist[j]

    parameters["G0"].value = G0
    parameters["Gw"].value = Gw
    parameters["Gu"].value = Gu

    # compute D and W
    D = np.zeros((nx * N, 1))
    W = np.zeros((N * nx, N * nx))

    for i in range(N):
        D[i * nx : (i + 1) * nx, :] = dlist[i]
        W[i * nx : (i + 1) * nx, i * nx : (i + 1) * nx] = Wlist[i]

    parameters["D"].value = D
    parameters["W"].value = W

    # compute Rs
    S = block_diag(Sigma_0, W)
    u, s, _ = np.linalg.svd(S, hermitian=True)
    Rs = u @ np.diag(np.sqrt(s))
    parameters["Rs"].value = Rs

    # compute Qbig and Rbig
    # take only diagonal elements of Qlist and Rlist
    qdiag = np.zeros(nx * (N + 1))
    rdiag = np.zeros(nu * N)

    for i in range(N):
        qdiag[i * nx : (i + 1) * nx] = np.diag(Qlist[i])
        rdiag[i * nu : (i + 1) * nu] = np.diag(Rlist[i])
    qdiag[N * nx :] = np.diag(Qlist[N])
    parameters["Qbig"].value = np.diag(qdiag)
    parameters["Rbig"].value = np.diag(rdiag)

    # compute RQbig and RRbig
    Qbig = np.diag(qdiag)
    Rbig = np.diag(rdiag)
    u, s, _ = np.linalg.svd(Qbig, hermitian=True)
    RQbig = u @ np.diag(np.sqrt(s))
    parameters["RQbig"].value = RQbig

    u, s, _ = np.linalg.svd(Rbig, hermitian=True)
    RRbig = u @ np.diag(np.sqrt(s))
    parameters["RRbig"].value = RRbig

    # set Xref and Uref
    for i in range(N + 1):
        parameters["Xref"][i].value = Xref[i]
    for i in range(N):
        parameters["Uref"][i].value = Uref[i]

    # set ObsAvoidConst
    for i in range(len(ObsAvoidConst)):
        A, B = ObsAvoidConst[i]
        parameters[f"A_obst_{i}"].value = A
        parameters[f"B_obst_{i}"].value = B

    # # if some parameters for obstacles were not set, set them to zero
    # for i in range(len(ObsAvoidConst), 10):
    #     parameters[f"A_obst_{i}"].value = np.zeros((nx * (N + 1), 2))
    #     parameters[f"B_obst_{i}"].value = np.zeros((2, 2))

    # set mu_0
    parameters["mu_0"].value = mu_0

    prob.solve(solver=cp.CLARABEL, verbose=False)

    return (
        variables["uffvar"].value,
        variables["Lvar"].value,
        variables["Kvar"].value,
        prob.status,
    )


if __name__ == "__main__":
    nx = 4
    nu = 2
    N = 10
    N_obs = 5
    prob, params = formulate_problem(nx, nu, N, N_obs=N_obs)

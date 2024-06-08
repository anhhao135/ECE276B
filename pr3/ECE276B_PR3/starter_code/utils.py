import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from matplotlib import animation
from time import time
from tqdm import tqdm
from casadi import *

x_init = 1.5
y_init = 0.0
theta_init = np.pi / 2
v_max = 1
v_min = 0
w_max = 1
w_min = -1
time_step = 0.5  # time between steps in seconds
T = 100
sim_time = 120  # simulation time
sigma = np.array([0.04, 0.04, 0.004])


# This function returns the reference point at time step k
def lissajous(k):
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2 * np.pi / (T * time_step)
    b = 3 * a
    k = k % T
    delta = np.pi / 2
    xref = xref_start + A * np.sin(a * k * time_step + delta)
    yref = yref_start + B * np.sin(b * k * time_step)
    v = [A * a * np.cos(a * k * time_step + delta), B * b * np.cos(b * k * time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]


# This function implements a simple P controller
def simple_controller(cur_state, ref_state):
    k_v = 0.5
    k_w = 1
    v = k_v * np.sqrt((cur_state[0] - ref_state[0]) ** 2 + (cur_state[1] - ref_state[1]) ** 2)
    v = np.clip(v, v_min, v_max)
    angle_diff = ref_state[2] - cur_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    w = k_w * angle_diff
    w = np.clip(w, w_min, w_max)
    return [v, w]

def getError(currentState, referenceState):
    return currentState - referenceState

def getPosition(state):
    return state[0:2]

def getOrientation(state):
    return state[2]

def errorMotionModelNoNoise(delta_t, p_err, theta_err, u_t, currRefState, nextRefState):
    #u_t_2d = np.atleast_2d(u_t).T
    u_t_2d = np.array([[u_t[0]],[u_t[1]]])
    #p_err_2d = np.atleast_2d(p_err).T
    p_err_2d = np.array([[p_err[0]],[p_err[1]],[theta_err]])
    G = np.array([[delta_t * np.cos(theta_err + currRefState[2]), 0], [delta_t * np.sin(theta_err + currRefState[2]), 0], [0, delta_t]])
    refPosDiff = np.atleast_2d(np.array(currRefState[0:2]) - np.array(nextRefState[0:2])).T
    refOriDiff = np.atleast_2d(np.array(currRefState[2] - nextRefState[2]))
    refDiff = np.vstack((refPosDiff,refOriDiff))
    p_err_next = (p_err_2d + G @ u_t_2d + refDiff).flatten()
    return vertcat(p_err_next[0], p_err_next[1], p_err_next[2])




def NLP_controller(delta_t, horizon, traj, currentIter, currentState, freeSpaceBounds, obstacle1, obstacle2):

    obstacleCenter = np.array([[obstacle1[0], obstacle1[1]]]).T
    obstacleRadius = obstacle1[2]

    obstacleCenter2 = np.array([[obstacle2[0], obstacle2[1]]]).T
    obstacleRadius2 = obstacle2[2]

    referenceStatesAhead = []

    for i in range(currentIter, currentIter + horizon + 1):
        referenceStatesAhead.append(traj(i))


    U = MX.sym('U', 2 * horizon)
    P = MX.sym('E', 2 * horizon + 2)
    Theta = MX.sym('theta', horizon + 1)
    E_given = getError(currentState, referenceStatesAhead[0])

    lowerBoundv = 0
    upperBoundv = 10
    lowerBoundw = -10
    upperBoundw = 10

    lowerBoundPositionError = -5
    upperBoundPositionError = 5

    lowerBoundOrientationError = -0.3
    upperBoundOrientationError = 0.3

    controlInputSolverLowerConstraint = list(np.tile(np.array([lowerBoundv, lowerBoundw]), horizon))
    controlInputSolverUpperConstraint = list(np.tile(np.array([upperBoundv, upperBoundw]), horizon))

    positionErrorSolverLowerConstraint = list(lowerBoundPositionError * np.ones(2 * horizon + 2))
    positionErrorSolverUpperConstraint = list(upperBoundPositionError * np.ones(2 * horizon + 2))

    orientationErrorSolverLowerConstraint = list(lowerBoundOrientationError * np.ones(horizon + 1))
    orientationErrorSolverUpperConstraint = list(upperBoundOrientationError * np.ones(horizon + 1))

    initialConditionSolverConstraint = list(np.zeros(5 * horizon + 3))
    motionModelSolverConstraint = list(np.zeros((horizon+1) * 3))


    upperBoundObstacleAvoider = 100

    obstacleSolverLowerConstraint = list(np.zeros(2 * horizon))
    obstacleSolverUpperConstraint = list(upperBoundObstacleAvoider * np.ones(2 * horizon))


    mapBoundSolverLowerConstraint = list(np.tile(np.array([freeSpaceBounds[0], freeSpaceBounds[1]]), horizon))
    mapBoundSolverUpperConstraint = list(np.tile(np.array([freeSpaceBounds[2], freeSpaceBounds[3]]), horizon))


    variables = vertcat(U, P, Theta)

    Q = 4 * np.eye(2)
    QBatch = np.kron(np.eye(horizon,dtype=int),Q)

    R = 2 * np.eye(2)
    RBatch = np.kron(np.eye(horizon,dtype=int),R)

    q = 1

    gammaValue = 0.95
    gammas = np.zeros(horizon)
    for i in range(horizon):
        gammas[i] = gammaValue**i

    gammas2D = np.eye(2 * horizon)
    for i in range(horizon):
        gammas2D[i*2:i*2+2,i*2:i*2+2] = gammas[i] * np.eye(2)


    costFunction = (P[2*(horizon-1)]**2 + P[2*(horizon-1) + 1]**2 + Theta[horizon-1]**2) + P[:2*horizon].T @ gammas2D @ QBatch @ P[:2*horizon] + U.T @ gammas2D @ RBatch @ U + q * np.atleast_2d(gammas) @ (1 - cos(Theta[:horizon]))**2

    motionModelConstraint0 = vertcat(P[0:2], Theta[0]) - E_given

    g = vertcat(motionModelConstraint0)

    for i in range(horizon):
        motionModelConstraint = vertcat(P[(i+1)*2:(i+1)*2+2], Theta[i+1]) - errorMotionModelNoNoise(delta_t, P[i*2:i*2+2], Theta[i], U[i*2:i*2+2], referenceStatesAhead[i], referenceStatesAhead[i+1])
        g = vertcat(g, motionModelConstraint)

    for i in range(horizon): #obstacle 1
        d = P[(i+1)*2:(i+1)*2+2] + referenceStatesAhead[i+1][0:2] - obstacleCenter
        g = vertcat(g, d.T @ d - (obstacleRadius+0.2)**2) 
    
    for i in range(horizon): #obstacle 2
        d = P[(i+1)*2:(i+1)*2+2] + referenceStatesAhead[i+1][0:2] - obstacleCenter2
        g = vertcat(g, d.T @ d - (obstacleRadius2+0.2)**2) 

    for i in range(horizon): #map bounds
        d = P[(i+1)*2:(i+1)*2+2] + referenceStatesAhead[i+1][0:2]
        g = vertcat(g, d)

    solver_params = {
    "ubg": motionModelSolverConstraint + obstacleSolverUpperConstraint + mapBoundSolverUpperConstraint,
    "lbg" :motionModelSolverConstraint + obstacleSolverLowerConstraint + mapBoundSolverLowerConstraint,
    "lbx": controlInputSolverLowerConstraint + positionErrorSolverLowerConstraint + orientationErrorSolverLowerConstraint,
    "ubx": controlInputSolverUpperConstraint + positionErrorSolverUpperConstraint + orientationErrorSolverUpperConstraint,
    "x0": initialConditionSolverConstraint
    }

    opts = {'ipopt.print_level':0, 'print_time':0}
    solver = nlpsol("solver", "ipopt", {'x':variables, 'f':costFunction, 'g':g}, opts)


    sol = solver(**solver_params)
    return [float(sol["x"][0]), float(sol["x"][1])]



# This function implement the car dynamics
def car_next_state(time_step, cur_state, control, noise=True):
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    w_xy = np.random.normal(0, sigma[0], 2)
    w_theta = np.random.normal(0, sigma[2], 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step * f.flatten() + w
    else:
        return cur_state + time_step * f.flatten()


def visualize(car_states, ref_traj, obstacles, t, time_step, save=False):
    init_state = car_states[0, :]

    def create_triangle(state=[0, 0, 0], h=0.5, w=0.25, update=False):
        x, y, th = state
        triangle = np.array([[h, 0], [0, w / 2], [0, -w / 2], [h, 0]]).T
        rotation_matrix = np.array([[cos(th), -sin(th)], [sin(th), cos(th)]])

        coords = np.array([[x, y]]) + (rotation_matrix @ triangle).T
        if update == True:
            return coords
        else:
            return coords[:3, :]

    def init():
        return (
            path,
            current_state,
            target_state,
        )

    def animate(i):
        # get variables
        x = car_states[i, 0]
        y = car_states[i, 1]
        th = car_states[i, 2]

        # update path
        if i == 0:
            path.set_data(np.array([]), np.array([]))
        x_new = np.hstack((path.get_xdata(), x))
        y_new = np.hstack((path.get_ydata(), y))
        path.set_data(x_new, y_new)

        # update horizon
        # x_new = car_states[0, :, i]
        # y_new = car_states[1, :, i]
        # horizon.set_data(x_new, y_new)

        # update current_state
        current_state.set_xy(create_triangle([x, y, th], update=True))

        # update current_target
        x_ref = ref_traj[i, 0]
        y_ref = ref_traj[i, 1]
        th_ref = ref_traj[i, 2]
        target_state.set_xy(create_triangle([x_ref, y_ref, th_ref], update=True))

        # update target_state
        # xy = target_state.get_xy()
        # target_state.set_xy(xy)

        return (
            path,
            current_state,
            target_state,
        )

    circles = []
    for obs in obstacles:
        circles.append(plt.Circle((obs[0], obs[1]), obs[2], color="r", alpha=0.5))
    # create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    min_scale_x = min(init_state[0], np.min(ref_traj[:, 0])) - 1.5
    max_scale_x = max(init_state[0], np.max(ref_traj[:, 0])) + 1.5
    min_scale_y = min(init_state[1], np.min(ref_traj[:, 1])) - 1.5
    max_scale_y = max(init_state[1], np.max(ref_traj[:, 1])) + 1.5
    ax.set_xlim(left=min_scale_x, right=max_scale_x)
    ax.set_ylim(bottom=min_scale_y, top=max_scale_y)
    for circle in circles:
        ax.add_patch(circle)
    # create lines:
    #   path
    (path,) = ax.plot([], [], "k", linewidth=2)

    #   current_state
    current_triangle = create_triangle(init_state[:3])
    current_state = ax.fill(current_triangle[:, 0], current_triangle[:, 1], color="r")
    current_state = current_state[0]
    #   target_state
    target_triangle = create_triangle(ref_traj[0, 0:3])
    target_state = ax.fill(target_triangle[:, 0], target_triangle[:, 1], color="b")
    target_state = target_state[0]

    #   reference trajectory
    ax.scatter(ref_traj[:, 0], ref_traj[:, 1], marker="x")

    sim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=len(t),
        interval=time_step * 100,
        blit=True,
        repeat=True,
    )
    plt.show()

    if save == True:
        sim.save("./fig/animation" + str(time()) + ".gif", writer="ffmpeg", fps=15)

    return


def timer(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        tqdm.write(f"{func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func

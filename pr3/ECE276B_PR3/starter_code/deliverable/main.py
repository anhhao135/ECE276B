from time import time
import numpy as np
import utils
import my_utils
import sys

def main():

    if len(sys.argv) == 1:
        raise Exception("Need to specify either nlp or gpi controller!")
    
    typeOfController = str(sys.argv[1])

    if not (typeOfController == "nlp" or typeOfController == "gpi"):
        raise Exception("Need to specify either nlp or gpi controller!")
    
    plotTitle = ""

    #initialize tuning parameters
    Q = 0
    R = 0
    q = 0
    gamma = 0
    CEC_horizon = 0

    simulationIterations = 200

    policy = None
    traj = utils.lissajous
    # Obstacles in the environment
    obstacles = np.array([[-2, -2, 0.5], [1, 2, 0.5]])

    if (typeOfController == "nlp"):
        #nlp tuning parameters
        Q = 20
        R = 5
        q = 5
        gamma = 0.9
        CEC_horizon = 8
    
    if (typeOfController == "gpi"):
        #gpi tuning parameters
        Q = 9
        R = 1
        q = 13
        gamma = 0.95
        obstaclePenalty = 50
        useExistingPolicy = True

        if useExistingPolicy:
            policy = np.loadtxt('policy.txt')
        else:
            policy = my_utils.GPI_CalculatePolicy(Q, R, q, gamma, traj, obstaclePenalty, obstacles[0], obstacles[1]) #calculate policy using parameters

    
    #GPI controller
    GPI_stateSpace = np.loadtxt('stateSpace.txt')
    GPI_controlSpace = np.loadtxt('controlSpace.txt')


    # Params
    ref_traj = []
    error_trans = 0.0
    error_rot = 0.0
    car_states = []
    times = []
    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([utils.x_init, utils.y_init, utils.theta_init])
    cur_iter = 0
    # Main loop
    #while cur_iter * utils.time_step < utils.sim_time:
    while cur_iter < simulationIterations:
        t1 = time()
        # Get reference state
        cur_time = cur_iter * utils.time_step
        cur_ref = traj(cur_iter)
        print(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        control = None

        if typeOfController == "nlp":
            control = my_utils.NLP_controller(utils.time_step, CEC_horizon, traj, cur_iter, cur_state, [-3,-3,3,3], obstacles[0], obstacles[1], Q, R, q, 0.2, gamma)
        elif typeOfController == "gpi":
            control = my_utils.GPI_controller(cur_iter, cur_state, GPI_stateSpace, GPI_controlSpace, traj, policy)

        print("Current ref state", cur_ref)
        print("Current robot state", cur_state)
        print("Current control input", control)
        ################################################################

        # Apply control input
        next_state = utils.car_next_state(utils.time_step, cur_state, control, noise=True)
        # Update current state
        cur_state = next_state 
        # Loop time
        t2 = utils.time()
        times.append(t2 - t1)
        cur_err = cur_state - cur_ref
        cur_err[2] = np.arctan2(np.sin(cur_err[2]), np.cos(cur_err[2]))
        error_trans = error_trans + np.linalg.norm(cur_err[:2])
        error_rot = error_rot + np.abs(cur_err[2])
        #print(cur_err, error_trans, error_rot)
        #print("======================")
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print("\n\n")
    print("Total time: ", main_loop_time - main_loop)
    print("Average iteration time: ", np.array(times).mean() * 1000, "ms")
    print("Final total translational error: ", error_trans)
    print("Final total rotational error: ", error_rot)

    if (typeOfController == "nlp"):
        plotTitle = "CEC, NLP-solved, trajectory tracking\nQ: {Q}, R: {R}, q: {q}, gamma: {gamma}, horizon: {CEC_horizon}\nTotal translational error: {transError}, total rotational error: {rotError}\nSimulation iterations: {simIter}, average iteration time: {avgIterTime} ms".format(Q = Q, R = R, q = q, gamma = gamma, CEC_horizon = CEC_horizon, transError = error_trans, rotError = error_rot, simIter = simulationIterations, avgIterTime = np.array(times).mean() * 1000)
    if (typeOfController == "gpi"):
        plotTitle = "GPI trajectory tracking\nQ: {Q}, R: {R}, q: {q}, gamma: {gamma}\nTotal translational error: {transError}, total rotational error: {rotError}\nSimulation iterations: {simIter}, average iteration time: {avgIterTime} ms".format(Q = Q, R = R, q = q, gamma = gamma, transError = error_trans, rotError = error_rot, simIter = simulationIterations, avgIterTime = np.array(times).mean() * 1000)

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    utils.visualize(car_states, ref_traj, obstacles, times, utils.time_step, False, plotTitle)


if __name__ == "__main__":
    main()


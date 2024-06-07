from casadi import *
import numpy as np
import utils
from my_utils import *

while False:
    Q = np.array([[1,2],[3,4]])
    Q = np.kron(np.eye(2,dtype=int),Q)
    print(Q)

    P = MX.sym('P', 4, 1)
    f = P.T @ Q @ P
    print(f)


while False:

    # Symbols/expressions
    x = MX.sym('x')
    y = MX.sym('y')
    z = MX.sym('z')
    f = x**2+100*z**2
    g = z+(1-x)**2-y

    nlp = {}                 # NLP declaration
    nlp['x']= vertcat(x,y,z) # decision vars
    nlp['f'] = f             # objective
    nlp['g'] = vertcat(g,g2)             # constraints

    # Create solver instance
    F = nlpsol('F','ipopt',nlp)

    # Solve the problem using a guess
    sol = F(x0=[2.5,3.0,0.75],ubg=0,lbg=0)

    print(sol["x"])



delta_t = 1
horizon = 2 #how many steps are looked ahead in receding horizon CEC, including the current step
positionVectorSize = 2
errorVectorSize = 3
controlVectorSize = 2
traj = utils.lissajous
currentIter = 0
w= np.random.normal(0, 0.1, 3)
currentState = traj(currentIter) + w

gammaValue = 0.9
gamma = np.zeros(horizon-1)
for i in range(horizon-1):
    gamma[i] = gammaValue**i

referenceStatesAhead = []

for i in range(currentIter + horizon):
    referenceStatesAhead.append(traj(i))

exampleControl = np.array([0.4, 0.5])
exampleError = getError(currentState, referenceStatesAhead[0])
exampleErrorMotionModel = errorMotionModelNoNoise(delta_t, exampleError, exampleControl, referenceStatesAhead[0], referenceStatesAhead[1])


# Initialize problem
U = MX.sym('U', 2)
E = MX.sym('E', 3)
E_given = np.array([0.1, 0.1, 0.01])

variables = vertcat(U, E)

costFunction = E[0]**2 + E[1]**2 + E[2]**2 + gamma[0] * (E_given[0]**2 + E_given[1]**2 + (1 - np.cos(E_given[2]))**2 + U[0]**2 + U[1]**2)

motionModelConstraint1 = E[0:3] - errorMotionModelNoNoise(delta_t, E_given, U[0:2], referenceStatesAhead[0], referenceStatesAhead[1])

solver_params = {
"ubg": [0],
"lbg" :[0],
"lbx": [-2,-2,-5,-5,-5],
"ubx": [2,2,5,5,5],
"x0": [0,0,0.1,0.1,0.01]
}
# Solve
solver = nlpsol("solver", "ipopt", {'x':variables, 'f':costFunction, 'g':motionModelConstraint1}, {})
sol = solver(**solver_params)
print("-----")
print("objective at solution = " + str(sol["f"]))
print("primal solution = " + str(sol["x"]))

print(errorMotionModelNoNoise(delta_t, E_given, np.array([0.138, 0.00565]), referenceStatesAhead[0], referenceStatesAhead[1]))


while False:


    # Initialize problem
    x = SX.sym('x')
    y = SX.sym('y')

    variables = vertcat(x, y)

    # Objective function
    # Min: -(x^2 + y^2)
    f = -(x**2 + y**2)

    # Constraints
    # x + y <= 10
    # x <= 5
    # 0 <= x,y <= 100
    #g = vertcat(x*y)
    #g = vertcat(np.array([[x,y]]) @ np.array([[x,y]]).T, x)

    solver_params = {
    "ubg": [10 , 5],
    "lbx": [-10, 10],
    "ubx": [10000, 10000],
    "x0": [2, 2]
    }
    # Solve
    solver = nlpsol("solver", "ipopt", {'x':variables, 'f':f, 'g':g}, {})
    sol = solver(**solver_params)

    # Print solution
    print("-----")
    print("objective at solution = " + str(sol["f"]))
    print("primal solution = " + str(sol["x"]))


while False:

    def errorMotionModelNoNoise(delta_t, p_err_t, theta_err_t, u_t, r_t, r_t1, a_t, a_t1):
        e_t = np.vstack((p_err_t, theta_err_t))
        G = np.array([[delta_t * np.cos(e_t[2,0] + a_t), 0], [delta_t * np.sin(e_t[2,0] + a_t), 0], [0, delta_t]])
        e_t1 = e_t + G @ u_t + np.vstack((r_t - r_t1, a_t - a_t1))
        return [e_t1[0:2,0], e_t1[2,0]]
        

    #result = errorMotionModelNoNoise(1, np.array([1,2]), 3, np.array([[1,2]]).T, np.array([[1,2]]).T, np.array([[3,5]]).T, 0.5, 0.6)
    #print(result)

    delta_t = 1

    horizon = 2 #how many steps are looked ahead in receding horizon CEC

    positionVectorSize = 2
    errorVectorSize = 3
    controlVectorSize = 2

    Q = np.eye(positionVectorSize) #baseline position tracking error stage cost scaler
    q = 1 #baseline orientation tracking error stage cost scaler
    R = np.eye(controlVectorSize) #baseline excessive control effort stage cost scaler

    r = np.array([[1,2], [3,4]])
    a = np.array([0, 0.1])
    print(r[0,:].shape)





    gammaValue = 0.9
    gamma = np.zeros((horizon-1,1))
    for i in range(horizon-1):
        gamma[i,0] = gammaValue**i

    P_err = MX.sym('P_err', positionVectorSize * horizon)
    Theta_err = MX.sym('Theta_err', horizon)
    U = MX.sym('U', controlVectorSize * horizon)

    #print(P_err[0:2,0].shape)

    errorMM0 = P_err.T
    print(errorMM0.shape)
    #g0 = p_err[0:2] - 

    #g = vertcat(x[0] + x[1] , x[0])

    X = MX.sym('X')
    Y = MX.sym('Y')
    g = np.array([X, Y])

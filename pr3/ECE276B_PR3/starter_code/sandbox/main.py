from casadi import *
import numpy as np

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

while False:
    # Initialize problem
    x = SX.sym('x', 2)

    # Objective function
    # Min: -(x^2 + y^2)
    f = -(x[0]**2 + x[1]**2)

    # Constraints
    # x + y <= 10
    # x <= 5
    # 0 <= x,y <= 100
    g = vertcat(x[0] + x[1] , x[0])
    solver_params = {
    "ubg": [10, 5],
    "lbx": [0, 0],
    "ubx": [100, 100],
    "x0": [2, 2]
    }
    # Solve
    solver = nlpsol("solver", "ipopt", {'x':x, 'f':f, 'g':g}, {})
    sol = solver(**solver_params)

    # Print solution
    print("-----")
    print("objective at solution = " + str(sol["f"]))
    print("primal solution = " + str(sol["x"]))


def errorMotionModelNoNoise(delta_t, e_t, u_t, r_t, r_t1, a_t, a_t1):
    G = np.array([[delta_t * np.cos(e_t[2,0] + a_t), 0], [delta_t * np.sin(e_t[2,0] + a_t), 0], [0, delta_t]])
    e_t1 = e_t + G @ u_t + np.vstack((r_t - r_t1, a_t - a_t1))
    return e_t1 
     

delta_t = 1
e_t = np.array([[1,2,3]]).T
u_t = np.array([[1,2]]).T
r_t = np.array([[1,2]]).T
r_t1 = np.array([[3,5]]).T
a_t = 0.5
a_t1 = 0.6

result = errorMotionModelNoNoise(delta_t, e_t, u_t, r_t, r_t1, a_t, a_t1)
print(result)


while False:

    horizon = 3 #how many steps are looked ahead in receding horizon CEC

    positionVectorSize = 2
    errorVectorSize = 3
    controlVectorSize = 2

    Q = np.eye(positionVectorSize) #baseline position tracking error stage cost scaler
    q = 1 #baseline orientation tracking error stage cost scaler
    R = np.eye(controlVectorSize) #baseline excessive control effort stage cost scaler

    gammaValue = 0.9
    gamma = np.zeros((horizon-1,1))
    for i in range(horizon-1):
        gamma[i,0] = gammaValue**i

    e = SX.sym('e', errorVectorSize * horizon)

import cvxpy as cp
import numpy as np

V = cp.Variable(4) #create optimal value function variable
objective = cp.Maximize(cp.sum(V)) #object to maximize sum of value functions
gamma = 0.9
constraints = [V[0] <= -10 + gamma*V[1], V[0] <= -8 + gamma*(0.5*V[0] + 0.5*V[2]), V[1] <= 1 + gamma*V[2], V[1] <= 8 + gamma*V[1], V[2] <= 1 + gamma*(0.25*V[0] + 0.75*V[1]), V[2] <= 2 + gamma*(0.5*V[2] + 0.5*V[3]), V[3] <= 6 + gamma*(0.5*V[0] + 0.5*V[3]), V[3] <= 3 + gamma*V[1]]
#constraints derived from linear programming
prob = cp.Problem(objective, constraints) #create optimzation problem
result = prob.solve() #optimize
print(V.value) #print out optimal value functions


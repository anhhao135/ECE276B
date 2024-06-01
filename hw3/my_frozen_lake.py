import numpy as np

class MDP(object):
  def __init__(self, P, nX, nU, gamma = 0.95):
    self.nX = nX
    self.nU = nU
    self.gamma = gamma
    self.P = np.zeros((nX,nU,nX)) # transition probability: SxAxS' -> [0,1]
    self.L = np.zeros((nX,nU))    # stage cost: SxA -> R
    self.Y = np.full(nX,np.inf)   # terminal cost: S -> R       
    for x in range(nX):
      for u in range(nU):
        for (pr,nx,reward,done) in P[x][u]:
          self.P[x,u,nx] += pr
          self.L[x,u] -= reward*pr
          if done: self.Y[nx] = 0



#controls: north (0), east (1), south (2), west (3)

stateSpaceSize = 25
controlSpaceSize = 4

L = np.zeros((25,4)) #initialize all stage costs to 0
#now populate the stage cost

#first do out of map costs
L[0:5,0] = 1 #top row move north would bring it out of map, incurs cost of 1
L[[4,9,14,19,24],1] = 1 #rightmost column move east would bring it out of map, incurs cost of 1
L[20:25,2] = 1 #bottom row move south would bring it out of map, incurs cost of 1
L[[0,5,10,15,20],3] = 1 #leftmost column move west would bring it out of map, incurs cost of 1

#second do special costs
L[1,:] = -10 #special state A, all controls incur -10
L[3,:] = -5 #special state B, all controls incur -5


P = np.zeros((stateSpaceSize,controlSpaceSize,stateSpaceSize)) #initialize motion model matrix: P[state t,control input, state t+1] = probability of occurence

#first we populate in P what happens for control input 0: move north
for i in range(5):
    P[i,0,i] = 1 #for the top row, moving north would determininstically put it in same spot
for i in range(5,25):
    P[i,0,i-5] = 1 #for other rows, moving north would determininstically put it in cell above

#second we populate in P what happens for control input 1: move east
for i in [4,9,14,19,24]:
    P[i,1,i] = 1 #for the right column, moving east would determininstically put it in same spot
for i in range(25):
    if i not in [4,9,14,19,24]:
        P[i,1,i+1] = 1 #for other columns, moving east would determininstically put it in cell to the right

#third we populate in P what happens for control input 2: move south
for i in range(20,25):
    P[i,2,i] = 1 #for the bottom row, moving south would determininstically put it in same spot
for i in range(20):
    P[i,2,i+5] = 1 #for other rows, moving south would determininstically put it in cell below

#fourth we populate in P what happens for control input 3: move west
for i in [0,5,10,15,20]:
    P[i,3,i] = 1 #for the left column, moving west would determininstically put it in same spot
for i in range(25):
    if i not in [0,5,10,15,20]:
        P[i,3,i-1] = 1 #for other columns, moving west would determininstically put it in cell to the left

#last we populate in P what happens for the special states

#clear the motion model from above since these are special cases
P[1] = 0
P[3] = 0

P[1,:,21] = 1 #all controls from special state A (1) will take it to state A' (21) deterministically
P[3,:,13] = 1 #all controls from special state B (3) will take it to state B' (13) deterministically


gamma = 0.9 #discount factor


iterations = 1000
V = np.zeros((iterations+1, stateSpaceSize))
pi = np.zeros((iterations+1,stateSpaceSize),dtype='int')
Q = np.zeros((iterations+1, stateSpaceSize, controlSpaceSize))


for k in range(iterations):
    Q_k = L + gamma * np.sum(P * np.min(Q[k,:],axis=1), axis=2)
    #print(Q_k.shape)
    pi[k+1,:] = np.argmin(Q_k, axis=1) #policy improvement
    Q[k+1,:,:] = Q_k #value update
    #print(pi[k+1])
    #print(Q[k+1])
    #maxValueDifference = np.abs(V[k+1] - V[k]).max()
    #print(maxValueDifference)
print(pi[-1])

for i in range(pi[-1].shape[0]):
    print(Q[-1,i,pi[-1,i]])

#print(V[-1])





while False:

    for k in range(iterations):
        P_pi = np.zeros((stateSpaceSize, stateSpaceSize))
        for i in range(P_pi.shape[0]):
            for j in range(P_pi.shape[1]):
                P_pi[i,j] = P[i,pi[k,i],j]

        L_pi = np.zeros(stateSpaceSize)
        for i in range(L_pi.shape[0]):
            L_pi[i] = L[i,pi[k,i]]

        A = np.eye(stateSpaceSize) - gamma * P_pi
        b = L_pi
        V[k] = np.linalg.solve(A, b)


        Q = L + gamma * np.sum(P * V[k,:], axis=2)
        pi[k+1,:] = np.argmin(Q, axis=1) #policy improvement

        print(V[k])
        print(pi[k])



while False:


    for k in range(iterations):
        Q = L + gamma * np.sum(P * V[k,:], axis=2)
        pi[k+1,:] = np.argmin(Q, axis=1) #policy improvement
        V[k+1,:] = np.min(Q, axis=1) #value update
        maxValueDifference = np.abs(V[k+1] - V[k]).max()
        print(maxValueDifference)
    print(pi[-1])
    print(V[-1])


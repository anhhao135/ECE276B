from dp_utils import *
from minigrid.core.world_object import Goal, Key, Door, Wall
import numpy as np

def getObjectType(object):
    if isinstance(object, type(None)):
        return "none"
    elif isinstance(object, Goal):
        return "goal"
    elif isinstance(object, Key):
        return "key"
    elif isinstance(object, Door):
        return "door"
    elif isinstance(object, Wall):
        return "wall"

def getTypeInFront(env):
    frontObject = env.grid.get(env.front_pos[0],env.front_pos[1])
    return getObjectType(frontObject)

def getTypeLeft(env):
    forwardDirectionVector = env.dir_vec
    leftDirectionVector = np.array([forwardDirectionVector[1],-forwardDirectionVector[0]])
    leftCell = env.agent_pos + leftDirectionVector
    leftObject = env.grid.get(leftCell[0],leftCell[1])
    return getObjectType(leftObject)

def getTypeRight(env):
    forwardDirectionVector = env.dir_vec
    rightDirectionVector = np.array([-forwardDirectionVector[1],forwardDirectionVector[0]])
    rightCell = env.agent_pos + rightDirectionVector
    rightObject = env.grid.get(rightCell[0],rightCell[1])
    return getObjectType(rightObject)

def checkIfDeadEnd(env):
    if (getTypeInFront(env) == "wall" and getTypeRight(env) == "wall" and getTypeLeft(env) == "wall"):
        return True
    else:
        return False
    
def getCurrentState(env, envInfo):
    agentPosition = env.agent_pos
    agentDirection = env.dir_vec
    door = env.grid.get(envInfo["door_pos"][0], envInfo["door_pos"][1])
    doorIsOpenStatus = door.is_open
    keyPickedUpStatus = env.carrying is not None
    stateVector = np.array([agentPosition[0],agentPosition[1],agentDirection[0],agentDirection[1],int(doorIsOpenStatus),int(keyPickedUpStatus)])
    return stateVector


def returnPossibleNextStates(env):
    print(env.front_pos)
    print(env.grid.get(env.front_pos[0]))
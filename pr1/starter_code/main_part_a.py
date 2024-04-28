from utils import *
from lib_part_a import *
import warnings
warnings.filterwarnings('ignore')

env_path = "./envs/known_envs/doorkey-5x5-normal.env"
env, info = load_env(env_path)  # load an environment
timeHorizon = 1000
seq = calculateOptimalSequence(env, info, timeHorizon)
draw_gif_from_seq(seq, env)  # draw a GIF & save 
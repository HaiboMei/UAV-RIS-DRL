#-- coding:UTF-8 --
"""
Double DQN & Natural DQN comparison,
The Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""
from RIS_UAV_env import RIS_UAV
from RL_brain import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
from Res_plot import Res_plot

env = RIS_UAV()
res = Res_plot(env)
MEMORY_SIZE = 3200
Episodes = env.eps

sess = tf.Session()
with tf.variable_scope('Double_DQN_UAV'):
    double_DQN_UAV = DoubleDQN(
        n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True,  ris=False, passive_shift=False, sess=sess, output_graph=True)

with tf.variable_scope('Double_DQN_FIX_RIS'):
    double_DQN_RIS_Fix = DoubleDQN(
        n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, ris=True, passive_shift=False, sess=sess, output_graph=True)

with tf.variable_scope('Double_DQN_RIS'):
    double_DQN_RIS = DoubleDQN(
        n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, ris=True, passive_shift=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())

# record the results
UAV_trajectory_ris = np.zeros((Episodes, env.N_slot, 3), dtype=np.float)
GT_schedule_ris= np.zeros((Episodes, env.N_slot), dtype=np.float)
UAV_flight_time_ris =  np.zeros((Episodes, env.N_slot), dtype=np.float)
slot_ris = np.zeros((1, Episodes), dtype=np.int)

UAV_trajectory_ris_no_shift = np.zeros((Episodes, env.N_slot, 3), dtype=np.float)
GT_schedule_ris_no_shift= np.zeros((Episodes, env.N_slot), dtype=np.float)
UAV_flight_time_ris_no_shift =  np.zeros((Episodes, env.N_slot), dtype=np.float)
slot_ris_no_shift = np.zeros((1, Episodes), dtype=np.int)

UAV_trajectory_no_ris = np.zeros((Episodes, env.N_slot, 3), dtype=np.float)
GT_schedule_no_ris = np.zeros((Episodes, env.N_slot), dtype=np.float)
UAV_flight_time_no_ris = np.zeros((Episodes, env.N_slot), dtype=np.float)
slot_no_ris = np.zeros((1, Episodes), dtype=np.int)

def train(RL):
    total = 0
    for ep in range(Episodes):
        observation = env.reset()

        slot = 0
        #for slot in range(env.N_slot):
        while (env.finish==False):
            action_index = RL.choose_action(observation)
            action = env.find_action(action_index)
            observation_, reward = env.step(action,slot,RL.ris,RL.passive_shift)
            RL.store_transition(observation, action, reward, observation_)

            if (RL.ris == True)&(RL.passive_shift==True):
                UAV_trajectory_ris[ep,slot,:] = observation_[:]
                GT_schedule_ris[ep,slot] = action[-2:-1]
                UAV_flight_time_ris[ep,slot] = action[-1:]
            if (RL.ris == True)&(RL.passive_shift==False):
                UAV_trajectory_ris_no_shift[ep,slot, :] = observation_[:]
                GT_schedule_ris_no_shift[ep, slot] = action[-2:-1]
                UAV_flight_time_ris_no_shift[ep, slot] = action[-1:]
            if (RL.ris == False):
                UAV_trajectory_no_ris[ep, slot, :] = observation_[:]
                GT_schedule_no_ris[ep, slot] = action[-2:-1]
                UAV_flight_time_no_ris[ep,slot] = action[-1:]

            if total+slot >= MEMORY_SIZE:
                RL.learn()
            observation = observation_
            slot = slot +1
            total = total+1
            if (RL.ris == True) & (RL.passive_shift == True):
                slot_ris[0,ep]= slot_ris[0,ep]+1
            if (RL.ris == True) & (RL.passive_shift == False):
                slot_ris_no_shift[0, ep] = slot_ris_no_shift[0, ep] + 1
            if (RL.ris == False):
                slot_no_ris[0, ep] = slot_no_ris[0, ep] + 1
        print("Finish episode %d" %ep)
        if (RL.ris == True) & (RL.passive_shift == True):
            UAV_trajectory_ris[ep,:]=env.UAV_FLY(UAV_trajectory_ris[ep,:],slot_ris[0,ep])
        if (RL.ris == True) & (RL.passive_shift == False):
            UAV_trajectory_ris_no_shift[ep, :] = env.UAV_FLY(UAV_trajectory_ris_no_shift[ep, :],slot_ris_no_shift[0,ep])
        if (RL.ris == False):
            UAV_trajectory_no_ris[ep, :] = env.UAV_FLY(UAV_trajectory_no_ris[ep, :],slot_no_ris[0,ep])
    return RL.q

print("Double_DQN_RIS")
train(double_DQN_RIS)
print("Double_DQN_FIX_RIS")
train(double_DQN_RIS_Fix)
print("Double_DQN_UAV")
train(double_DQN_UAV)

EPS=env.eps-1
res.plot_UAV_GT(env.w_k,UAV_trajectory_ris,UAV_trajectory_ris_no_shift,UAV_trajectory_no_ris,env.N_slot,slot_ris,slot_ris_no_shift,slot_no_ris,UAV_flight_time_ris,UAV_flight_time_ris_no_shift,UAV_flight_time_no_ris)
res.plot_propulsion_energy(UAV_trajectory_ris,UAV_trajectory_ris_no_shift,UAV_trajectory_no_ris,UAV_flight_time_ris,UAV_flight_time_ris_no_shift,UAV_flight_time_no_ris,EPS,slot_ris,slot_ris_no_shift,slot_no_ris)
res.plot_data_throughput(UAV_trajectory_ris,UAV_trajectory_ris_no_shift,UAV_trajectory_no_ris,GT_schedule_ris,GT_schedule_ris_no_shift,GT_schedule_no_ris,UAV_flight_time_ris,UAV_flight_time_ris_no_shift,UAV_flight_time_no_ris,EPS,slot_ris,slot_ris_no_shift,slot_no_ris)
res.plot_energy_efficiency(UAV_trajectory_ris,UAV_trajectory_ris_no_shift,UAV_trajectory_no_ris,GT_schedule_ris,GT_schedule_ris_no_shift,GT_schedule_no_ris,UAV_flight_time_ris,UAV_flight_time_ris_no_shift,UAV_flight_time_no_ris,EPS,slot_ris,slot_ris_no_shift,slot_no_ris)
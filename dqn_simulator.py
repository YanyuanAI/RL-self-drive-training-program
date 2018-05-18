#!/usr/bin/env python
from __future__ import print_function

import argparse

import random
import threading
from collections import deque

import cv2
import eventlet
import eventlet.wsgi
import numpy as np
import socketio
import tensorflow as tf
from flask import Flask
import time

time_dqn = time.time()
time_skt = time.time()
time_tem=time_dqn
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

GAME = 'car'

ACTIONS = 2

GAMMA = 0.99  # decay rate of past observations
OBSERVE = 1000.  # timesteps to observe before training
EXPLORE = 3000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1
terminal1=0
curr=0
is_start=0
reward=0
throttle = 12
steering_angle = 0
con = 0
connect=0
flag=0
x_t = cv2.imread("c:/MainCamera.png")
x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
ret, x_t = cv2.threshold(x_t, 170, 255, cv2.THRESH_BINARY)
s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
s_t1=s_t

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")


def ImageToMatrix(im):
    width, height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(height,width))
    return new_data
def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    D = deque()
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1

    '''x_t = cv2.imread("c:/MainCamera.png")
    #x_t = np.asarray(s)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)

    ret, x_t = cv2.threshold(x_t,170,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)'''
    '''s_t是公有变量'''

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    """
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    """
    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
        #time_dqn=time.time()
        print("t:")
        print(t)
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing
        if a_t[0]==1:
            global steering_angle
            steering_angle = readout_t[0]*(-1)

        else:
            global steering_angle
            steering_angle = readout_t[1]


        #time_dqn=time.time()      #等待时间更新readout_t[1]**********************************


        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward

            '''print("is_start")
            print(is_start)
            print(curr)'''
        print("333333333333333333333333")
        print(steering_angle)
        print(time_dqn)
        print(time_skt)

        while (time_dqn>=time_skt):
            i = 1
        print("44444444444444444444444444444")
        print(time_dqn)
        print(time_skt)
        global time_dqn
        global time_tem
        time_dqn=time.time()        #等待时间更新***************************************************
        time_tem=time_dqn
        print("更新t1 前面")
        print(time_dqn)
        '''x_t1_colored= cv2.imread("c:/MainCamera.png")
        r_t=reward
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 170, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        #print (s_t.shape)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)'''

        # store the transition in D
        r_t=reward
        D.append((s_t, a_t, r_t, s_t1, terminal1))
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        i=0
        while(i<10000):
            i+=1

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)
            #print (minibatch.shape)
            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            #print (s_j_batch.shape)
            a_batch = [d[1] for d in minibatch]
            #print (a_batch.shape)
            r_batch = [d[2] for d in minibatch]
            #print (r_batch.shape)
            s_j1_batch = [d[3] for d in minibatch]
            #print (s_j1_batch.shape)

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )
        global time_dqn
        time_dqn = time.time()  # 等待时间更新s_t1**********************************
        print("更新t1 后面")
        print(time_dqn)

        # update the old values
        global s_t
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        '''print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))'''
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 12
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    print("telemetryyyyyyyyyyyyyyyyyyyyyyy")
    if data:
        global time_skt
        time_skt=time.time()
        # The current steering angle of the car
        #steering_angle = data["steering_angle"]
        # The current throttle of the car
        #throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        global reward
        reward = float(data["reward"])
        terminal = data["terminal"]
        if terminal=="1":
            global terminal1
            terminal1=1
        else:
            global terminal1
            terminal1=0

        global curr
        curr += 1
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


    # 获取image的np数组
    global x_t
    x_t = cv2.imread("c:/MainCamera.png")
    global x_t
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 170, 255, cv2.THRESH_BINARY)
    global x_t
    x_t = np.reshape(x_t, (80, 80, 1))
    global s_t1
    s_t1 = np.append(x_t, s_t[:, :, :3], axis=2)

    '''x_t1_colored= cv2.imread("c:/MainCamera.png")
          r_t=reward
          x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
          ret, x_t1 = cv2.threshold(x_t1, 170, 255, cv2.THRESH_BINARY)
          x_t1 = np.reshape(x_t1, (80, 80, 1))
          #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
          #print (s_t.shape)
          s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)'''
    print("11111111111111111111111111111111")
    print(time_dqn)
    print(time_tem)
    while (time_dqn == time_tem):
        r = 1
    print("2222222222222222222222222222222")
    global steering_angle,throttle
    send_control(steering_angle, throttle)
    print("telemetry")


@sio.on('connect')
def connect(sid, environ):
    global connect
    connect=1
    print("connect ", sid)
    #send_control(0, 0)
    cn()
    send_start()
    send_control(0, 0)



def cn():
    global con
    con=1

def send_control(steering_angle, throttle):

    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)
    print("senddddddddddddddddddddddddddd")
    print(steering_angle)
    print(throttle)
def send_start():
    global is_start
    is_start += 1
    sio.emit(
        "start",
        data={
            'is_start': is_start.__str__()
            },
        skip_sid=True)
    print("start")
    print(is_start)
    print(curr)

def loop():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()

    trainNetwork(s, readout, h_fc1, sess)
    print('thread %s ended.')
def loop2(app):
    print("dddddddddddddddddddddddd")
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


t = threading.Thread(target=loop, name='LoopThread')
t2 = threading.Thread(target=loop2,args=(app,), name='LoopThread2')
t.start()

t2.start()
t.join()
t2.join()
print('thread %s ended.')

import gymnasium as gym
import cv2
import tensorflow as tf

cartEnv = gym.make("CartPole-v1", render_mode="rgb_array")
NUM_EPISODES = 5
for episode in range(NUM_EPISODES):
    done = False
    state = cartEnv.reset()
    while not done:
        frame = cartEnv.render()
        cv2.imshow("CartPole", frame)
        cv2.waitKey(100)
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32).numpy()
        state, reward, done, _, _ = cartEnv.step(action)
cartEnv.close()
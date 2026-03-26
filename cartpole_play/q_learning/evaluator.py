import gymnasium as gym
import cv2
import tensorflow as tf
from keras.models import load_model

env = gym.make("CartPole-v1", render_mode="rgb_array")
q_net = load_model("q_learning_q_net.keras")

def policy(state, explore=0.0):
    action = tf.argmax(q_net(state)[0], output_type=tf.int32)
    if tf.random.uniform(shape=(), maxval=1) <= explore:
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)
    return action

for episode in range(5):
    done = False
    obs, _ = env.reset()
    state = tf.convert_to_tensor([obs])
    while not done:
        frame = env.render()
        cv2.imshow("CartPole", frame)
        cv2.waitKey(100)
        action = policy(state)
        state, reward, done, _, _ = env.step(action.numpy())
        state = tf.convert_to_tensor([state])
env.close()

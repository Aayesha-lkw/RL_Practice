import gymnasium as gym 
import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense
import pickle as pkl


env = gym.make("CartPole-v1")

net_input = Input(shape=(4,))
x = Dense(64, activation="relu")(net_input)
x = Dense(32, activation="relu")(x)
output = Dense(2, activation="linear")(x)
q_net = Model(inputs=net_input, outputs=output)

ALPHA = 0.001
EPSILON = 1.0
EPSILON_DECAY = 1.001
GAMMA = 0.99
NUM_EPISODES = 500

def policy(state, explore=0.0):
    action = tf.argmax(q_net(state)[0], output_type = tf.int32)
    if tf.random.uniform(shape=(), maxval=1) <= explore:
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)
    return action

for episode in range(NUM_EPISODES):
    done = False
    obs, _ = env.reset()
    state = tf.convert_to_tensor([obs])
    action = policy(state, EPSILON)
    total_rewards = 0
    episode_length = 0
    while not done:
        next_state, reward, done, _, _ = env.step(action.numpy())
        next_state = tf.convert_to_tensor([next_state])
        next_action = policy(next_state, EPSILON)
        target = reward + GAMMA * q_net(next_state)[0][next_action]
        if done:
            target = reward
        with tf.GradientTape() as Tape:
            current = q_net(state)
        grads = Tape.gradient(current, q_net.trainable_weights)
        delta = target - current[0][action]
        for j in range(len(grads)):
            q_net.trainable_weights[j].assign_add(ALPHA * delta * grads[j])
        state = next_state
        action = next_action
        total_rewards += reward
        episode_length += 1
    print("Episode: ", episode, "Length: ", episode_length, "Reward: ", total_rewards, "Epsilon: ", EPSILON)
    EPSILON /= EPSILON_DECAY
q_net.save("sarsa_q_net.keras")
env.close()




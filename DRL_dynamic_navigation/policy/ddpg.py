import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Conv2D, MaxPool2D, Reshape
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
# revise it from crowd_sim.py
lidar_dim = 1800
lidar_dim_short = 60
lidar_interval = 30
n_lidar = 5
position_dim = 2
action_dim = 2
# revise it from crowd_sim.py

def actor(units=(256, 128)):
    inputs = [Input(shape=(lidar_dim_short * n_lidar,)), Input(shape=(position_dim,))]
    concat = Concatenate(axis=-1)(inputs)
    x = Dense(units[0], name="L0", activation='relu')(concat)
    for index in range(1, len(units)):
        x = Dense(units[index], name="L{}".format(index), activation='relu')(x)
    output = Dense(action_dim, name="Out", activation='tanh')(x)
    model = Model(inputs=inputs, outputs=output)

    return model


def critic(units=(256, 128)):
    inputs = [Input(shape=(lidar_dim_short * n_lidar,)), Input(shape=(position_dim,)), Input(shape=(action_dim,))]
    concat = Concatenate(axis=-1)(inputs)
    x = Dense(units[0], name="L0", activation='relu')(concat)
    for index in range(1, len(units)):
        x = Dense(units[index], name="L{}".format(index), activation='relu')(x)
    output = Dense(1, name="Out")(x)
    model = Model(inputs=inputs, outputs=output)

    return model

def update_target_weights(model, target_model):
    weights = model.get_weights()
    target_model.set_weights(weights)

def shorten_lidar(lidar):
    return lidar[:, 0:7200:30]


class DDPG:
    def __init__(self, actor_units=(256, 256, 64),
                       critic_units=(256, 256, 64),):        
        self.lr = None
        self.action_dim = None
        self.model = None
        self.target_model = None
        self.gamma = None

        # Define and initialize Actor network
        self.actor = actor(actor_units)
        self.actor_target = actor(actor_units)
        update_target_weights(self.actor, self.actor_target)

        # Define and initialize Critic network
        self.critic = critic(critic_units)
        self.critic_target = critic(critic_units)
        update_target_weights(self.critic, self.critic_target)
        
    def configure(self, action_dim, gamma=0.95):
        self.gamma = gamma
        self.action_dim = action_dim

    def set_lr(self, lr=0.0001):
        self.lr = lr
        self.actor_optimizer = Adam(learning_rate=lr)
        self.critic_optimizer = Adam(learning_rate=lr)

    def target_update(self):
        update_target_weights(self.actor, self.actor_target)  # iterates target model
        update_target_weights(self.critic, self.critic_target)

    def optimize_batch(self, train_batches, memory, batch_size):
        for _ in range(train_batches):
            states1_long, states2, actions, rewards, next_states1_long, next_states2, dones = memory.sample(batch_size)
            states1 = shorten_lidar(states1_long)
            next_states1 = shorten_lidar(next_states1_long)
            next_actions = self.actor_target.predict([next_states1, next_states2])
            q_future = self.critic_target.predict([next_states1, next_states2, next_actions])
            target_qs = rewards + q_future * self.gamma * (1. - dones)

            # train critic
            with tf.GradientTape() as tape:
                q_values = self.critic([states1, states2, actions])
                td_error = q_values - target_qs
                critic_loss = tf.reduce_mean(tf.math.square(td_error))

            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)  # compute critic gradient
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

            # train actor
            with tf.GradientTape() as tape:
                actions = self.actor([states1, states2])
                actor_loss = -tf.reduce_mean(self.critic([states1, states2, actions]))

            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)  # compute actor gradient
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

    def get_action(self, lidar, position, decay=0.1): 
        state1_long = np.reshape(lidar, [1, lidar_dim * n_lidar])
        state1 = shorten_lidar(state1_long)
        state2 = np.reshape(position, [1, position_dim])
        a = self.actor.predict([state1, state2])
        noise = tf.random.normal([action_dim], 0.0, decay * 0.5)
        a += noise
        a = tf.clip_by_value(a, -1.0, 1.0)
        return a[0]

    def save_model(self, fn):
        a_fn = fn + '_actor.h5'
        c_fn = fn + '_critic.h5'
        self.actor.save(a_fn)
        self.critic.save(c_fn)

    def load_model(self, fn):
        a_fn = fn + '_actor.h5'
        c_fn = fn + '_critic.h5'
        self.actor.load_weights(a_fn)
        self.actor_target.load_weights(a_fn)
        self.critic.load_weights(c_fn)
        self.critic_target.load_weights(c_fn)

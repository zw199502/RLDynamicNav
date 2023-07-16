import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Conv2D, MaxPool2D, Reshape
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

tfd = tfp.distributions

# revise it from crowd_sim.py
lidar_dim = 1800
n_lidar = 5
position_dim = 2
action_dim = 2
encoder_out = 64
# revise it from crowd_sim.py

def encoder():
    lidar_input_shape = (lidar_dim * n_lidar,)
    lidar_input = Input(lidar_input_shape)
    lidar_input_reshape = Reshape((n_lidar, lidar_dim, 1), input_shape=lidar_input_shape)(lidar_input)
    conv1 = Conv2D(32, kernel_size=[2, 20], strides=(1, 10), activation='relu')(lidar_input_reshape)
    pool1 = MaxPool2D(pool_size=[1, 5])(conv1)
    conv2 = Conv2D(32, kernel_size=[2, 2],  strides=(1, 1), activation='relu')(pool1)
    pool2 = MaxPool2D(pool_size=[1, 2])(conv2)
    flatten = Flatten()(pool2)
    linear1 = Dense(128, activation='relu')(flatten)
    linear2 = Dense(128, activation='relu')(linear1)
    linear3 = Dense(encoder_out,  activation='relu')(linear2)
    return Model(lidar_input, linear3)

def actor(units=(400, 200, 100)):
    encoder_out_shape = (encoder_out,)
    encoder_input = Input(encoder_out_shape)
    position_input_shape = (position_dim,)
    position_input = Input(position_input_shape)
    concat_input = Concatenate(axis=-1)([encoder_input, position_input])
    x = Dense(units[0], name="L0", activation="relu")(concat_input)
    for index in range(1, len(units)):
        x = Dense(units[index], name="L{}".format(index), activation="relu")(x)

    action_shape = (action_dim, )
    actions_mean = Dense(action_shape[0], name="Out_mean")(x)
    actions_std = Dense(action_shape[0], name="Out_std")(x)

    model = Model(inputs=[encoder_input, position_input], outputs=[actions_mean, actions_std])

    return model


def critic(units=(400, 200, 100)):
    encoder_out_shape = (encoder_out,)
    encoder_input = Input(encoder_out_shape)
    position_input_shape = (position_dim,)
    position_input = Input(position_input_shape)
    action_shape = (action_dim, )
    action_input = Input(action_shape)
    concat_input = Concatenate(axis=-1)([encoder_input, position_input, action_input])
    x = Dense(units[0], name="Hidden0", activation="relu")(concat_input)
    for index in range(1, len(units)):
        x = Dense(units[index], name="Hidden{}".format(index), activation="relu")(x)

    output = Dense(1, name="Out_QVal")(x)
    model = Model(inputs=[encoder_input, position_input, action_input], outputs=output)

    return model

def update_target_weights(model, target_model):
    weights = model.get_weights()
    target_model.set_weights(weights)

class SAC:
    def __init__(self, actor_units=(128, 64),
                       critic_units=(128, 64),
                       auto_alpha=True,
                       alpha=0.2):

        self.gamma = None
        self.action_shape = (action_dim,)

        # models
        self.encoder = encoder()
        print(self.encoder.summary())

        # Define and initialize actor network
        self.actor = actor(actor_units)
        self.log_std_min = -20
        self.log_std_max = 2
        print(self.actor.summary())

        # Define and initialize critic networks
        self.critic_1 = critic(critic_units)
        self.critic_target_1 = critic(critic_units)
        update_target_weights(self.critic_1, self.critic_target_1)

        self.critic_2 = critic(critic_units)
        self.critic_target_2 = critic(critic_units)
        update_target_weights(self.critic_2, self.critic_target_2)

        print(self.critic_1.summary())


        # Define and initialize temperature alpha and target entropy
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -np.prod(self.action_shape)
            self.log_alpha = tf.Variable(0., dtype=tf.float32)
            self.alpha = tf.Variable(0., dtype=tf.float32)
            self.alpha.assign(tf.exp(self.log_alpha))
        else:
            self.alpha = tf.Variable(alpha, dtype=tf.float32)


    def configure(self, action_discrete, gamma=0.95):
        self.gamma = gamma

    def set_lr(self, lr=0.0001):
        self.lr = lr
        self.actor_optimizer = Adam(learning_rate=lr)
        self.critic_optimizer_1 = Adam(learning_rate=lr)
        self.critic_optimizer_2 = Adam(learning_rate=lr)
        self.alpha_optimizer = Adam(learning_rate=lr)
        self.encoder_optimizer = Adam(learning_rate=lr)

    def target_update(self):
        update_target_weights(self.critic_1, self.critic_target_1)  # iterates target model
        update_target_weights(self.critic_2, self.critic_target_2)

    def save_model(self, fn):
        a_fn = fn + '_actor.h5'
        c_fn = fn + '_critic.h5'
        e_fn = fn + '_encoder.h5'
        self.actor.save(a_fn)
        self.critic_1.save(c_fn)
        self.encoder.save(e_fn)

    def load_model(self, fn):
        a_fn = fn + '_actor.h5'
        c_fn = fn + '_critic.h5'
        e_fn = fn + '_encoder.h5'
        self.actor.load_weights(a_fn)
        self.critic_1.load_weights(c_fn)
        self.critic_target_1.load_weights(c_fn)
        self.critic_2.load_weights(c_fn)
        self.critic_target_2.load_weights(c_fn)
        self.encoder.load_weights(e_fn)

    def process_actions(self, mean, log_std, test=False, eps=1e-6):
        std = tf.math.exp(log_std)
        raw_actions = mean

        if not test:
            raw_actions += tf.random.normal(shape=mean.shape, dtype=tf.float32) * std

        log_prob_u = tfd.Normal(loc=mean, scale=std).log_prob(raw_actions)
        actions = tf.math.tanh(raw_actions)

        log_prob = tf.reduce_sum(log_prob_u - tf.math.log(1 - actions ** 2 + eps))

        return actions, log_prob

    def get_action(self, lidar, position, test=False, use_random=False):
        state1 = np.reshape(lidar, [1, lidar_dim * n_lidar])
        state2 = np.reshape(position, [1, position_dim])

        if use_random:
            a = tf.random.uniform(shape=(1, self.action_shape[0]), minval=-1, maxval=1, dtype=tf.float32)
        else:
            enc = self.encoder(state1)
            means, log_stds = self.actor.predict([enc, state2])
            log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)
            a, _ = self.process_actions(means, log_stds, test=test)

        return a[0]

    def optimize_batch(self, train_batches, memory, batch_size):
        for _ in range(train_batches):
            states1, states2, actions, rewards, next_states1, next_states2, dones = memory.sample(batch_size)
            
            with tf.GradientTape(persistent=True) as tape:
                # next state action log probs
                next_enc = tf.stop_gradient(self.encoder(next_states1))
                enc = self.encoder(states1)
                
                means, log_stds = self.actor([next_enc, next_states2])
                log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)
                next_actions, log_probs = self.process_actions(means, log_stds)

                # critics loss
                
                current_q_1 = self.critic_1([enc, states2, actions])
                current_q_2 = self.critic_2([enc, states2, actions])
                next_q_1 = self.critic_target_1([next_enc, next_states2, next_actions])
                next_q_2 = self.critic_target_2([next_enc, next_states2, next_actions])
                next_q_min = tf.math.minimum(next_q_1, next_q_2)
                state_values = next_q_min - self.alpha * log_probs
                target_qs = tf.stop_gradient(rewards + state_values * self.gamma * (1. - dones))
                critic_loss_1 = tf.reduce_mean(0.5 * tf.math.square(current_q_1 - target_qs))
                critic_loss_2 = tf.reduce_mean(0.5 * tf.math.square(current_q_2 - target_qs))
                critic_loss_sum = critic_loss_1 + critic_loss_2

                # current state action log probs
                means, log_stds = self.actor([enc, states2])
                log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)
                actions, log_probs = self.process_actions(means, log_stds)

                # actor loss
                current_q_1 = self.critic_1([enc, states2, actions])
                current_q_2 = self.critic_2([enc, states2, actions])
                current_q_min = tf.math.minimum(current_q_1, current_q_2)
                actor_loss = tf.reduce_mean(self.alpha * log_probs - current_q_min)

                # temperature loss
                if self.auto_alpha:
                    alpha_loss = -tf.reduce_mean(
                        (self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy)))

            
            encoder_grad = tape.gradient(critic_loss_sum, self.encoder.trainable_variables)  # compute encoder gradient
            self.encoder_optimizer.apply_gradients(zip(encoder_grad, self.encoder.trainable_variables))

            critic_grad = tape.gradient(critic_loss_1, self.critic_1.trainable_variables)  # compute actor gradient
            self.critic_optimizer_1.apply_gradients(zip(critic_grad, self.critic_1.trainable_variables))

            critic_grad = tape.gradient(critic_loss_2, self.critic_2.trainable_variables)  # compute critic_1 gradient
            self.critic_optimizer_2.apply_gradients(zip(critic_grad, self.critic_2.trainable_variables))

            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)  # compute critic_2 gradient
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))


            if self.auto_alpha:
                # optimize temperature
                alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
                self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
                self.alpha.assign(tf.exp(self.log_alpha))

        


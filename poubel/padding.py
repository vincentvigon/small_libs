# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
#
# pp = print
#
#
# def periodic_padding(W, pad: int):
#
#     left = W[:, :pad, :]
#     right = W[:, -pad:, :]
#     return tf.concat([right, W, left], axis=1)
#
#
# def zero_padding(W, pad):
#     s = W.shape
#     border = tf.zeros([s[0], pad, s[2]])
#     return tf.concat([border, W, border], axis=1)
#
#
# def dirichlet_padding(W, pad):
#     right_value = W[:, -1, :]
#     left_value = W[:, 0, :]
#     s = W.shape
#     left_value_repeat = tf.ones([s[0], pad, s[2]]) * left_value[:, tf.newaxis, :]
#     right_value_repeat = tf.ones([s[0], pad, s[2]]) * right_value[:, tf.newaxis, :]
#     return tf.concat([left_value_repeat, W, right_value_repeat], axis=1)
#
#
# def neumann_padding_4D_asym(W, nb_to_add):
#     if nb_to_add%2==0:
#         pad_left=nb_to_add//2
#         pad_right=nb_to_add//2
#     else:
#         pad_left = nb_to_add // 2
#         pad_right = nb_to_add // 2+1
#
#     right_value = W[:,:, -1, :]
#     left_value = W[:,:, 0, :]
#     s = W.shape
#
#     left_value_repeat = tf.ones([s[0],s[1], pad_left, s[3]]) * left_value[:,:, tf.newaxis, :]
#
#     right_value_repeat = tf.ones([s[0],s[1], pad_right, s[3]]) * right_value[:,:, tf.newaxis, :]
#
#     return tf.concat([left_value_repeat, W, right_value_repeat], axis=2)
#
#
# def activation_for_relexive():
#     x = tf.linspace(-3, 3, 100)
#     y = tf.nn.elu(x - 1) + 1
#     z = tf.nn.relu(x)
#     plt.plot(x, y)
#     plt.plot(x, z)
#     plt.show()
#
#
# # uniquement le reflexive quand on veut qu'il préserve la positivité
# def make_positive_channels_0_2(A):
#     A0, A1, A2 = A[:, :, 0], A[:, :, 1], A[:, :, 2]
#     A0 = tf.nn.elu(A0 - 1) + 1
#     A2 = tf.nn.elu(A2 - 1) + 1
#     res = tf.stack([A0, A1, A2], axis=2)
#     return res
#
#
# def reflexive_padding(W: tf.Tensor, pad: int, make_positive: bool):
#
#     right_value = W[:, -1, :]
#     left_value = W[:, 0, :]
#     s = W.shape
#     left_value_repeat = tf.ones([s[0], pad, s[2]]) * left_value[:, tf.newaxis, :]
#     right_value_repeat = tf.ones([s[0], pad, s[2]]) * right_value[:, tf.newaxis, :]
#
#     left = W[:, 1:pad + 1, :] - left_value_repeat
#     right = W[:, -1 - pad:-1, :] - right_value_repeat
#     left = -left[:, ::-1, :]
#     right = -right[:, ::-1, :]
#     left += left_value_repeat
#     right += right_value_repeat
#
#     if make_positive and make_positive:
#         left = make_positive_channels_0_2(left)
#         right = make_positive_channels_0_2(right)
#
#     return tf.concat([left, W, right], axis=1)
#
#
# def test_the_3_paddings():
#     x = tf.cast(np.linspace(0., 1, 110, endpoint=False), tf.float32)
#     y = tf.sin(x * 2 * np.pi) + (2 * x + 1)
#     W = tf.stack([y, y, y], axis=1)
#     W = W[tf.newaxis, :, :]
#
#     pad = 100
#     W_neumann = dirichlet_padding(W, pad)
#     W_periodic = periodic_padding(W, pad)
#     W_reflexive = reflexive_padding(W, pad, False)
#     W_reflexive_pos = reflexive_padding(W, pad, True)
#
#     assert W_neumann.shape == W_reflexive.shape == W_periodic.shape
#     plt.plot(W_neumann[0, :, 0], label="neumann")
#     plt.plot(W_periodic[0, :, 0], label="periodic")
#     plt.plot(W_reflexive[0, :, 0], label="reflexive")
#     plt.plot(W_reflexive_pos[0, :, 0], label="reflexive_pos")
#     plt.plot(np.zeros_like(x), color="k")
#     plt.legend()
#     plt.show()
#
#
# if __name__ == "__main__":
#     test_the_3_paddings()

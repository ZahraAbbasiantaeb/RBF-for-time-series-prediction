import random
import tensorflow as tf
from data_time_series import train_rbf_x, train_y,train_x, cluster_count, plot_scatter_two, train_ebf_x

epochs = 200
hidden_neurons = cluster_count


X = train_ebf_x

def init_weights(shape):

    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def feedForward_NN(x, w1, b1):

    Z = (tf.matmul(x, w1) + b1)

    return Z


tf.reset_default_graph()


tfX = tf.placeholder(tf.float32, [None, hidden_neurons])
tfY = tf.placeholder(tf.float32, [None])


w1 = init_weights([hidden_neurons, 1])
b1 = init_weights([1])

network = feedForward_NN(tfX, w1, b1)


with tf.name_scope("training"):

    # cost = tf.sqrt(tf.reduce_mean(tf.square(network- tfY)))
    cost = tf.reduce_mean(tf.losses.mean_squared_error((network[:,0]), tfY))

    optimizer = tf.train.AdamOptimizer(0.4, name='optimizer')

    train_op = optimizer.minimize(cost, name='train_op')


# train MSE
train_mse = cost

# validation MSE
validation_mse = tf.losses.mean_squared_error(network[:,0], tfY)


# add summaries
tf.summary.scalar('train_loss', cost)

merged_summary_op = tf.summary.merge_all()

step = 0

with tf.Session() as sess:

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    sess.run(init)

    writer = tf.summary.FileWriter("/graph")

    writer.add_graph(graph=tf.get_default_graph())

    for epoch in range(0, epochs):

        _, c, s = sess.run([train_op, cost, merged_summary_op], feed_dict={tfX:X, tfY:train_y}, )

        writer.add_summary(s, step)

        pred = sess.run(network, feed_dict={tfX:X, tfY:train_y})

        if(epoch%100 == 0):

            print(train_mse.eval({tfX: X, tfY: train_y}))
            print('******')

        step += 1

    save_path = saver.save(sess, "/model.ckpt")

    print("Model saved in path: %s" % save_path)

    plot_scatter_two(train_x,train_y, pred)

    print('done')

    print('train loss:')

    print(train_mse.eval({tfX: X, tfY: train_y}))






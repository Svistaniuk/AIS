import numpy as np
import matplotlib.pyplot as plt  
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

n_samples = 1000
batch_size = 100
num_steps = 20000
display_step = 2000

X_data = np.random.uniform(1, 10, (n_samples, 1))
y_data = 2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))

x = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

with tf.variable_scope('linear-regression'):
    k = tf.Variable(tf.random_normal((1, 1)), name='slope')
    b = tf.Variable(tf.zeros((1,)), name='bias')

y_pred = tf.matmul(x, k) + b

loss = tf.reduce_sum((y - y_pred) ** 2)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss)

final_k = 0
final_b = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("Початок навчання...")
    for i in range(num_steps):
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]

        _, loss_val, k_val, b_val = sess.run([optimizer, loss, k, b],
                                             feed_dict={x: X_batch, y: y_batch})

        if (i + 1) % display_step == 0:
            print(f'Епоха {i+1}: Loss = {loss_val:.4f}, k = {k_val[0][0]:.4f}, b = {b_val[0]:.4f}')
    
    final_k = k_val[0][0]
    final_b = b_val[0]
    print("\nНавчання завершено!")

plt.figure(figsize=(10, 6))

plt.scatter(X_data, y_data, s=8, label='Дані (з шумом)')

x_line = np.linspace(X_data.min(), X_data.max(), 100)
y_line = final_k * x_line + final_b

label_text = f'Лінійна регресія: y={final_k:.2f}x + {final_b:.2f}'
plt.plot(x_line, y_line, color='red', linewidth=3, label=label_text)

plt.title('Лінійна регресія з TensorFlow')
plt.xlabel('x')
plt.ylabel('y')
plt.legend() 
plt.grid(True, alpha=0.3) 

plt.show()
import  tensorflow as tf,numpy as np 

# //y=wx+b+noise,learning w ;
w_real=[0.2,0.3,0.6];
b_real=-0.2;
x_data=np.random.randn(2000,3);
noise=np.random.randn(1,2000)*0.1;

y_data=np.matmul(w_real,x_data.T)+b_real+noise;

STEP=20;
with tf.Graph().as_default():
    x_in=tf.placeholder(tf.float32,shape=(None,3));
    y_in=tf.placeholder(tf.float32,shape=None);
    w_train=tf.Variable(tf.zeros((1,3),dtype=tf.float32));
    b_train=tf.Variable(0,dtype=tf.float32);
    y_pred=tf.matmul(w_train,tf.transpose(x_in))+b_train;

    loss=tf.reduce_mean(tf.square(y_in-y_pred));
    learning_rate=0.3;
    optimizer=tf.train.GradientDescentOptimizer(learning_rate);
    train=optimizer.minimize(loss);

    init=tf.global_variables_initializer();
    with tf.Session() as sess:
        sess.run(init);
        for i in range(STEP):
            print(i);
            sess.run(train,{x_in:x_data,y_in:y_data})

            if i%5==0:
                print(sess.run([w_train,b_train]));

        pass


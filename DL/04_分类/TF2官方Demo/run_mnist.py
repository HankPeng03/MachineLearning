import tensorflow as tf
import os
import pdb
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    gpu = gpus[1] #如果有多个GPU，仅使用第0个GPU
    # tf.config.experimental.set_memory_growth(gpu0, True) #设置GPU显存用量按需使用
    # 或者也可以设置GPU显存为固定使用量(例如：4G)
    tf.config.set_visible_devices([gpu],"GPU")
    tf.config.experimental.set_virtual_device_configuration(gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

if __name__=="__main__":
    # 加载数据
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    # 添加channels维度
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    # 使用tf.data来将数据集切分为batch以及混淆数据集
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    # 使用Keras模型子类 API构建tf.keras模型
    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(filters=32, kernel_size=3, activation="relu")
            self.flatten = Flatten()
            self.d1 = Dense(units=128, activation="relu")
            self.d2 = Dense(units=10, activation="softmax")

        def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            x = self.d2(x)

    model = MyModel()

    # 为训练选择优化器与损失函数
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    # 选择衡量指标来度量模型的损失值与准确率
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name="train_accuracy")

    test_loss = tf.keras.metrics.Mean(name="train_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name="test_accuracy")

    # 使用tf.GradientTape来训练模型
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        pdb.set_trace()
        gradients = tape.gradient(target=loss, sources=model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 5
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        pdb.set_trace()
        for images, labels in train_ds:
            train_step(images, labels)

        for images, labels in test_ds:
            test_step(images, labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))
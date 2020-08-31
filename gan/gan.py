from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys
import os
import numpy as np

class GAN():
    def __init__(self):
        # --------------------------------- #
        #   行28，列28，也就是mnist的shape
        # --------------------------------- #
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        # 28,28,1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        #输入n维向量维度为100
        self.latent_dim = 100
        # adam优化器
        optimizer = Adam(0.0002, 0.5)

        #判别模型训练  当判别模型输出和真实输入图片类型不符时，用交叉熵训练
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        #生成模型训练  建立模型/输入/生成图片/传入判别模型，为0时训练
        self.generator = self.build_generator()
        gan_input = Input(shape=(self.latent_dim,))
        img = self.generator(gan_input)
        # 在训练generate的时候不训练discriminator
        self.discriminator.trainable = False
        # 传入判别模型，对生成的假图片进行预测
        validity = self.discriminator(img)
        self.combined = Model(gan_input, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):
        # ----------------------------------------- #
        #   生成器，输入一串随机数字n维向量，输出图片
        # ----------------------------------------- #
        model = Sequential()

        #输入n维向量，shape设为100维
        noise = Input(shape=(self.latent_dim,))  

        #全连接到256维的神经元上
        model.add(Dense(256, input_dim=self.latent_dim))
        #激活函数
        model.add(LeakyReLU(alpha=0.2))
        #标准化
        model.add(BatchNormalization(momentum=0.8))

        #256维神经元映射到512维神经元上
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        #512维神经元映射到1024维神经元上
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        #最终生成28x28x1，np.prod784维上
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        img = model(noise)

        #输出图片
        return Model(noise, img)

    def build_discriminator(self):
        # ----------------------------------- #
        #   评价器，对输入进来的图片进行评价
        # ----------------------------------- #
        model = Sequential()
        # 输入一张图片
        img = Input(shape=self.img_shape)

        #用Flatten将其平铺到784维向量
        model.add(Flatten(input_shape=self.img_shape))
        #映射到512维神经元，激活
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        #映射到256维神经元，激活        
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        # 判断真伪，全连接到一维向量，1维真，0为假
        model.add(Dense(1, activation='sigmoid'))

        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        # 获得数据
        (X_train, _), (_, _) = mnist.load_data()

        # 进行标准化，28,28->28,28,1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # 创建标签
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # --------------------------- #
            #   随机选取batch_size个图片
            #   对discriminator进行训练
            # --------------------------- #
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            #对真实图片生成noise
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            #假图片
            gen_imgs = self.generator.predict(noise)

            #判别网络对真假图片的判别
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            #  训练generator，与1比较，相差大则交叉熵大，得到训练
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    if not os.path.exists("./images"): 
        os.makedirs("./images")
    gan = GAN()
    gan.train(epochs=8001, batch_size=256, sample_interval=500)

from fourier_neural_operator.FNO_1d_plus import FNO1d_plus
from grid_up.grid_up_old import GridUp_testable, GridUp
from paddings.paddings import pad_nd, kind_dict
import time

from paddings.paddings_right import pad_nd_right
pp=print
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=10000,precision=4)



class DataCreator1D(GridUp_testable):


    def nus_intensities(self,batch_size):
        nus=tf.floor(tf.random.uniform(minval=0.5, maxval=5, shape=[batch_size, 1, 1]))
        intensities=tf.random.uniform(minval=0.1,maxval=1.,shape=[batch_size,1,1])
        return nus,intensities
    def generate_XY(self, batch_size) -> tuple:
        nus, intensities = self.nus_intensities(batch_size)
        X = self.X_fun(self.grid, nus) * intensities
        Y = self.Y_fun(self.grid, nus) * intensities
        return X, Y


    def plot_prediction(self, ax, model: tf.keras.Model) -> None:
        pass

    def score(self, Y_true, Y_pred) -> dict or float:
        pass

    def loss(self,Y_true,Y_pred):
        return tf.reduce_mean((Y_true - Y_pred) ** 2)

    def __init__(self,nx):
        self.nx=nx

        self.X_fun  = lambda x,nu: tf.sin(nu * 2 * np.pi * x ** 2)
        self.Y_fun = lambda x,nu: nu * tf.cos(nu * 2 * np.pi * x ** 2) * 2 * x
        #
        # self.A_fun = lambda x, nu: -tf.sin(nu * x)+0.5*tf.sin(2*nu * x)
        # self.U_fun = lambda x, nu: tf.cos(nu * x) ** 5 +0.3*tf.cos(2*nu * x)

        self.grid = tf.linspace(0., 1, self.nx)[tf.newaxis, :, tf.newaxis]



def test():
    resolutions_train=[100,150]
    resolutions_test=[50,75,100,125,150,175,200]
    resolutions_train = [100, 150]
    resolutions_test = [50, 75, 100, 125, 150, 175, 200]

    creator_train={nx:DataCreator1D(nx) for nx in resolutions_train}
    creator_test= {nx:DataCreator1D(nx) for nx in resolutions_test}

    fixed_params = {
        "first_channel_unchanged": False,
        "pad_prop": 0.1,
        "freq_mix_size": 0,
        "pad_kind": "zero_padding",
        "modes" : 20,
        "width" : 15,
        "nb_layer":4,
        "out_channels":1 #ne varie pas
    }


    pad_kinds = ["no_padding"]
    pad_kinds += kind_dict.keys()
    varying_params= {
        "first_channel_unchanged": [True,False],
        "pad_prop": [0,0.02,0.05,0.1,0.2],
        "freq_mix_size": [0,3,5,7,9],
        "pad_kind": pad_kinds,
        "modes" : [5,10,15,20,24],
        "width" : [5,10,15,20,25],
        "nb_layer":[2,3,4,5]
    }

    testor=GridUp(
        FNO1d_plus,
        creator_train,
        creator_test,
        fixed_params=fixed_params,
        varying_params=varying_params,
        minutes=0.01,
        verbose=True
    )

    testor.watch_param("first_channel_unchanged")
    testor.plot_last_result()
    testor.plot_prediction()



if __name__=="__main__":
    test()
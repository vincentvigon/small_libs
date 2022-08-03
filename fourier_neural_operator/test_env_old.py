# from fourier_neural_operator.FNO_1d_plus import FNO1d_plus
# from paddings.paddings import pad_nd, kind_dict
# import time
#
# from paddings.paddings_right import pad_nd_right
# pp=print
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# np.set_printoptions(linewidth=10000,precision=4)
#
#
#
# def default_param():
#     params = {
#         "first_channel_unchanged": False,
#         "pad_prop": 0.1,
#         "freq_mix_size": 0,
#         "pad_kind": "zero_padding",
#         "modes" : 20,
#         "width" : 15,
#         "nb_layer":4
#     }
#     return params
#
#
# class DataCreator1D:
#     def __init__(self,nx,batch_size):
#         self.nx=nx
#         self.batch_size=batch_size
#
#         self.X_fun  = lambda x,nu: tf.sin(nu * 2 * np.pi * x ** 2)
#         self.Y_fun = lambda x,nu: nu * tf.cos(nu * 2 * np.pi * x ** 2) * 2 * x
#         #
#         # self.A_fun = lambda x, nu: -tf.sin(nu * x)+0.5*tf.sin(2*nu * x)
#         # self.U_fun = lambda x, nu: tf.cos(nu * x) ** 5 +0.3*tf.cos(2*nu * x)
#
#         self.grid = tf.linspace(0., 1, self.nx)[tf.newaxis, :, tf.newaxis]
#
#     def nus_intensities(self):
#         nus=tf.floor(tf.random.uniform(minval=0.5, maxval=5, shape=[self.batch_size, 1, 1]))
#         intensities=tf.random.uniform(minval=0.1,maxval=1.,shape=[self.batch_size,1,1])
#         return nus,intensities
#
#     def call(self):
#         nus, intensities = self.nus_intensities()
#         X = self.X_fun(self.grid, nus) * intensities
#         Y = self.Y_fun(self.grid, nus) * intensities
#         return X,Y
#
#     def loss(self,Y_true,Y_pred):
#         return tf.reduce_sum((Y_true - Y_pred) ** 2) / self.nx / self.batch_size
#
#
# def test_changing_first_channel_unchanged():
#     first_channel_unchangeds = [False,True]
#     print("test les first_channel_unchanged dans:" + str(first_channel_unchangeds))
#     test_changing_something("first_channel_unchanged",first_channel_unchangeds)
#
# def test_changing_freq_mix_size():
#     freq_mix_sizes = [0,3,5,7,9]
#     print("test les freq_mix_size dans:" + str(freq_mix_sizes))
#     test_changing_something("freq_mix_size",freq_mix_sizes)
#
#
# def test_changing_pad_prop():
#     pad_props = [0,0.02,0.05,0.1,0.2]
#
#     print("test les pad_prop dans:" + str(pad_props))
#     test_changing_something("pad_prop",pad_props)
#
#
# def test_changing_pad_kind():
#     pad_kinds = ["no_padding"]
#     pad_kinds += kind_dict.keys()
#     print("test les pad_kind dans:" + str(pad_kinds))
#     test_changing_something("pad_kind",pad_kinds)
#
# def test_changing_modes():
#     modess = [5,10,15,20,24]
#     print("test les modes dans:" + str(modess))
#     test_changing_something("modes",modess)
#
# def test_changing_width():
#     widths = [5,10,15,20,25]
#     print("test les width dans:" + str(widths))
#     test_changing_something("width",widths)
#
#
# def test_changing_nb_layer():
#     nb_layers = [2,3,4,5]
#     print("test les nb_layer dans:" + str(nb_layers))
#     test_changing_something("nb_layer",nb_layers)
#
#
#
# def test_changing_something(key:str,values):
#     verbose=True
#     params=default_param()
#
#     resolutions_train=[100,150]
#     resolutions_test=[50,75,100,125,150,175,200]
#     data_creators_train=[DataCreator1D(nx,64) for nx in resolutions_train]
#     data_creators_test=[DataCreator1D(nx,64) for nx in resolutions_test]
#
#     fig,ax=plt.subplots(figsize=(5,5))
#     for val in values:
#         if verbose:
#             print(f"key={key}, val={val}")
#         params[key]=val
#         losses_test=test_one_params(data_creators_train,data_creators_test,FNO1d_plus,params,verbose)
#         if verbose:
#             print("losses_test",losses_test)
#
#         ax.plot(resolutions_test,losses_test,".-",label=str(val))
#
#         for x,y in zip(resolutions_test,losses_test):
#             if x in resolutions_train:
#                 ax.plot(x,y,".k")
#
#     ax.set_xticks(resolutions_test)
#     ax.set_yscale("log")
#     ax.set_ylabel("mse")
#     ax.set_xlabel("nx")
#     ax.set_title(key)
#     ax.legend()
#     plt.show()
#
#
#
# def test_one_params(data_creators_train,data_creators_test,ClassModel,params,verbose):
#
#     model = ClassModel(
#         modes=params["modes"],
#         width=params["width"],
#         out_channels=1,
#         nb_layer=params["nb_layer"],
#         first_channel_unchanged=params["first_channel_unchanged"],
#         pad_prop=params["pad_prop"],
#         freq_mix_size=params["freq_mix_size"],
#         pad_kind=params["pad_kind"])
#     optimizer=tf.keras.optimizers.Adam()
#
#     # Le modèle contient plusieurs layers. On inspecte le premier.
#     #model.convs[0].inspect_weiths()
#
#     losses=[]
#     best_loss=float("inf")
#     best_weights=None
#
#     minutes=1
#     count = 0
#
#     try:
#         OK=True
#         ti0=time.time()
#         while OK:
#             count+=1
#             OK=time.time()-ti0<minutes*60
#
#             mean_loss=tf.constant(0.)
#             for data_creator_train in data_creators_train:
#                 X,Y=data_creator_train.call()
#                 with tf.GradientTape() as tape:
#                     Y_pred=model.call(X)
#                     loss=data_creator_train.loss(Y,Y_pred)
#
#                 grad=tape.gradient(loss,model.trainable_variables)
#                 optimizer.apply_gradients(zip(grad,model.trainable_variables))
#
#                 mean_loss+=loss
#
#
#             mean_loss= loss.numpy()/len(data_creators_train)
#             if mean_loss<best_loss:
#                 best_loss=mean_loss
#                 best_weights=model.get_weights()
#                 if verbose:
#                     print(f"↘{best_loss:.1e}",end="")
#
#             losses.append(mean_loss)
#
#     except KeyboardInterrupt:
#         pass
#
#     if verbose:
#         print(f"\n nb epochs:{count} during {minutes} minutes")
#         #plt.plot(losses)
#         #plt.show()
#
#     if best_weights is not None:
#         model.set_weights(best_weights)
#
#     losses_test=[]
#     for data_creator_test in data_creators_test:
#         X, Y = data_creator_test.call()
#         Y_pred = model.call(X)
#         loss=data_creator_test.loss(Y, Y_pred).numpy()
#         losses_test.append(loss)
#
#     return losses_test
#
#
# if __name__=="__main__":
#     #test_changing_pad_kind()
#     test_changing_pad_prop()
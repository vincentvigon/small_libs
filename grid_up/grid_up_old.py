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
# from abc import ABC, abstractmethod
# from typing import List,Dict,Type
# from copy import deepcopy
#
#
# class GridUp_testable(ABC):
#
#     @abstractmethod
#     def generate_XY(self,batch_size)->tuple:
#         pass
#
#     @abstractmethod
#     def plot_prediction(self,ax,model:tf.keras.Model)->None:
#         pass
#
#
#     @abstractmethod
#     def loss(self,Y_true,Y_pred)->tf.Tensor:
#         pass
#
#     @abstractmethod
#     def score(self, model) -> dict or float:
#         # Facultatif
#         # For plots
#         # To override if necessary
#         # Result can be a dictionnary if you need several score to be ploted
#         pass
#
#
# # def some_fixed_param():
# #     params = {
# #         "first_channel_unchanged": False,
# #         "pad_prop": 0.1,
# #         "freq_mix_size": 0,
# #         "pad_kind": "zero_padding",
# #         "modes" : 20,
# #         "width" : 15,
# #         "nb_layer":4,
# #         "out_channels" : 1
# #     }
# #     return params
# #
# #
# # def some_varying_params():
# #     pad_kinds = ["no_padding"]
# #     pad_kinds += kind_dict.keys()
# #
# #     params = {
# #         "first_channel_unchanged": [False,True],
# #         "pad_prop": [0,0.02,0.05,0.1,0.2],
# #         "freq_mix_size": [0,3,5,7,9],
# #         "pad_kind": pad_kinds,
# #         "modes" : [5,10,15,20,24],
# #         "width" : [5,10,15,20,25],
# #         "nb_layer":[2,3,4,5]
# #     }
# #     return params
#
# class GridUp:
#
#     def __init__(self,
#             ClassModel:Type[tf.keras.Model],
#             data_creators_train_dict:Dict[float,GridUp_testable],
#             data_creators_test_dict:Dict[float,GridUp_testable],
#             fixed_params:dict,
#             varying_params:dict,
#             minutes,
#             train_batch_size=64,
#             test_batch_size = 1024,
#             verbose=False):
#
#         self.data_creators_train_dict=data_creators_train_dict
#         self.data_creators_test_dict=data_creators_test_dict
#         self.fixed_params=fixed_params
#         self.varying_params=varying_params
#         self.train_batch_size = train_batch_size
#         self.test_batch_size = test_batch_size
#         self.verbose=verbose
#         self.minutes=minutes
#
#         self.ClassModel=ClassModel
#
#         one_data_gen = list(self.data_creators_train_dict.values())[0]
#         self.score_names=["loss"]
#         self.score_is_dict = None
#
#         if hasattr(one_data_gen, "score"):
#             model_bidon=self.ClassModel(**self.fixed_params)
#             one_score = one_data_gen.score(model_bidon)
#             if one_score is not None:
#                 self.score_is_dict = isinstance(one_score, dict)
#                 if self.score_is_dict:
#                     self.score_names+=list(one_score.keys())
#                 else:
#                     self.score_names.append("score")
#
#
#     def watch_param(self,paramName:str):
#         paramValues=self.varying_params[paramName]
#
#         print(f"Look for model.{paramName}={paramValues}. Train with data-cases:{list(self.data_creators_train_dict.keys())}, Test with data-cases:{list(self.data_creators_test_dict.keys())}")
#         print(f"Other parameters are:",[(k,v) for k,v in self.fixed_params.items() if k!=paramName])
#
#         all_params=deepcopy(self.fixed_params)
#
#         #resolutions_train=self.data_creators_train_dict.keys()#[100,150]
#         #resolutions_test=self.data_creators_test_dict.keys()#[50,75,100,125,150,175,200]
#
#         scoreName_paramVal_case_scoreVal={k:{} for k in self.score_names}
#
#         self.last_models={}
#         for paramVal in paramValues:
#             if self.verbose:
#                 print(f"key={paramName}, paramVal={paramVal}")
#             all_params[paramName]=paramVal
#
#             scoreName_case_val,model=self.train_and_test(all_params)
#             self.last_models[paramVal]=model
#
#             if self.verbose:
#                 print("scoreName_case_val:\n",scoreName_case_val)
#
#             for scoreName in self.score_names:
#                 scoreName_paramVal_case_scoreVal[scoreName][paramVal]=scoreName_case_val[scoreName]
#
#         self.last_scoreName_paramVal_case_scoreVal=scoreName_paramVal_case_scoreVal
#         self.last_paramName=paramName
#
#         return scoreName_paramVal_case_scoreVal
#
#
#     def plot_prediction(self):
#
#         # y_labels=[]
#         # y_cases=[]
#         # for case in self.data_creators_test_dict.keys():
#         #     label=str(case)
#         #     if self.data_creators_train_dict.get(case) is not None:
#         #         label+=" trained"
#         #     y_labels.append(label)
#         #     y_cases.append(case)
#         #
#         paramValues=self.varying_params[self.last_paramName]
#
#         ni,nj=len(self.data_creators_test_dict),len(paramValues)
#
#         fig,axs=plt.subplots(ni,nj,figsize=(4*nj,2*ni))
#         if ni==1:
#             axs=axs[None,:]
#         elif nj==1:
#             axs=axs[:,None]
#
#         for i,(case,data_creator) in enumerate(self.data_creators_test_dict.items()):
#             label_i=str(case)
#             if self.data_creators_train_dict.get(case) is not None:
#                 label_i+=" (trained)"
#             for j,paramValue in enumerate(paramValues):
#                 ax=axs[i,j]
#                 data_creator.plot_prediction(ax,self.last_models[paramValue])
#                 if j==0:
#                      ax.set_ylabel(label_i)
#                 if i==ni-1:
#                     ax.set_xlabel(str(paramValue))
#                 if i==j==0:
#                     ax.legend()
#
#         fig.text(0.5, 0.01, f"model.{self.last_paramName}", ha='center',fontsize=18)
#         fig.text(0.01, 0.5, f"cases", va='center', rotation='vertical',fontsize=18)
#
#         fig.tight_layout(w_pad=1, h_pad=1)
#
#         # hide tick and tick label of the big axis
#         # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
#         # plt.xlabel("common X")
#         # plt.ylabel("common Y")
#
#         plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)
#
#         return fig,axs
#
#
#     def plot_last_result(self,x_are_cases=True,yscale_log=True):
#
#         nb=len(self.score_names)
#         fig,axs=plt.subplots(nb,1,figsize=(10,5*nb))
#         if nb==1:
#             axs=[axs]
#
#         if x_are_cases:
#             plot_func=plot_x_are_cases
#         else:
#             plot_func=plot_x_are_params
#
#         for i,scoreName in enumerate(self.score_names):
#             plot_func(axs[i],scoreName,self.last_scoreName_paramVal_case_scoreVal[scoreName],self.last_paramName,list(self.data_creators_train_dict.keys()),yscale_log)
#
#         fig.tight_layout()
#
#         return fig,axs
#
#
#     def train_and_test(self,params):
#
#         model = self.ClassModel(**params)
#         optimizer=tf.keras.optimizers.Adam()
#
#         best_loss=float("inf")
#         best_weights=None
#         count = 0
#
#         try:
#             OK=True
#             ti0=time.time()
#             while OK:
#                 count+=1
#                 OK=time.time()-ti0<self.minutes*60
#                 mean_loss=0.
#
#                 for data_creator_train in self.data_creators_train_dict.values():
#                     X,Y=data_creator_train.generate_XY(self.train_batch_size)
#                     with tf.GradientTape() as tape:
#                         Y_pred=model.call(X)
#                         loss=data_creator_train.loss(Y,Y_pred)
#
#                     grad=tape.gradient(loss,model.trainable_variables)
#                     optimizer.apply_gradients(zip(grad,model.trainable_variables))
#
#                     mean_loss+=loss.numpy()
#
#
#                 mean_loss/=len(self.data_creators_train_dict)
#
#                 if mean_loss<best_loss:
#                     best_loss=mean_loss
#                     best_weights=model.get_weights()
#                     if self.verbose:
#                         print(f"↘{best_loss:.1e}",end="")
#
#         except KeyboardInterrupt:
#             pass
#
#         if self.verbose:
#             print(f"\n nb epochs:{count} during {self.minutes} minutes")
#
#         if best_weights is not None:
#             model.set_weights(best_weights)
#
#         return self.test(model),model
#
#
#     def test(self,model):
#
#         scoreName_case_val={score_name:{} for score_name in self.score_names}
#
#         for dataCase,data_creator in self.data_creators_test_dict.items():
#             X, Y = data_creator.generate_XY(self.test_batch_size)
#             Y_pred = model.call(X)
#             scoreName_case_val["loss"][dataCase]=data_creator.loss(Y,Y_pred).numpy()
#
#             if self.score_is_dict is not None:
#                 score = data_creator.score(model)
#                 if self.score_is_dict:
#                     for score_name in score.keys():
#                         scoreName_case_val[score_name][dataCase] = score[score_name]
#                 else:
#                     scoreName_case_val["score"][dataCase] = score
#
#         return scoreName_case_val
#
#
# def plot_x_are_cases(ax,scoreName,param_case_score,paramName,train_cases,yscale_log):
#     """
#     res: param → (case → score)
#     """
#
#     one_case_score=list(param_case_score.values())[0]
#     cases=list(one_case_score.keys())
#     cases_are_str=isinstance(cases[0],str)
#
#     cases_dico=None
#     train_cases_are_str=isinstance(train_cases[0],str)
#
#     if cases_are_str!=train_cases_are_str:
#         raise Exception("test and train cases are not of same type (str or scalar)")
#
#
#     if cases_are_str:
#         cases_num=range(len(cases))
#         cases_str=cases
#         cases_dico={k:v for k,v in zip(cases_str,cases_num)}
#     else:
#         cases_num=cases
#         cases_str=cases
#
#     for param,dico in param_case_score.items():
#         ax.plot(cases_num,list(dico.values()),".-",label=str(param))
#
#     for x in train_cases:
#         if cases_are_str:
#             num_val=cases_dico.get(x)
#             if num_val is not None:
#                 ax.axvline(x=num_val,linestyle=":")
#             else:
#                 print(f"warning: the train-case '{x}' cannot be represented because it is a string that not belongs to the test-cases")
#         else:
#             ax.axvline(x=x,linestyle=":")
#
#     ax.set_title(f"data_case→{scoreName} by {paramName}")
#     if yscale_log:
#         ax.set_yscale("log")
#     ax.set_xticks(cases_num)
#     ax.set_xticklabels(cases_str)
#     ax.set_xlabel("data cases")
#     ax.set_ylabel(scoreName)
#     ax.legend()
#
#
# def inverse_dico(param_case_score):
#     res={}
#     for param,case_score in param_case_score.items():
#         for case,score in case_score.items():
#             line=res.get(case,{})
#             line[param]=score
#             res[case]=line
#     return res
#
#
# def plot_x_are_params(ax,scoreName,param_case_score,paramName,train_cases,yscale_log):
#
#     case_param_score=inverse_dico(param_case_score)
#
#     param_scores=list(case_param_score.values())[0]
#
#     params=list(param_scores.keys())
#     params_are_str=isinstance(params[0],str)
#
#     if params_are_str:
#         params_num=range(len(params))
#         params_str=params
#     else:
#         params_num=params
#         params_str=params
#
#     train_cases=set(train_cases)
#
#     for case,param_score in case_param_score.items():
#         label=str(case)
#         if case in train_cases:
#             label+=" (trained)"
#         ax.plot(params_num,list(param_score.values()),".-",label=label)
#
#     ax.set_title(f"{paramName}→{scoreName} by data_case")
#     if yscale_log:
#         ax.set_yscale("log")
#     ax.set_xticks(params_num)
#     ax.set_xticklabels(params_str)
#     ax.set_xlabel(f"{paramName}")
#     ax.set_ylabel(scoreName)
#     ax.legend()
#
#
# """
# ##############################################  TEST ########################################@
# """
#
# class SimpleModelClass(tf.keras.Model):
#     def __init__(self,width,nb_layers):
#         super().__init__()
#
#         self.lays=[tf.keras.layers.Dense(width,activation="relu") for _ in range(nb_layers)]
#         self.final=tf.keras.layers.Dense(1)
#
#     def call(self,X):
#         for lay in self.lays:
#             X=lay(X)
#         return self.final(X)
#
#
#
#
# class DataCreator_num(GridUp_testable):
#
#     def __init__(self,nu):
#         self.nu=nu
#
#     def plot_prediction(self,ax,model):
#         X,Y=self.generate_XY(100)
#         Y_pred=model(X)
#         ax.plot(X[:,0],Y[:,0],".",label="true")
#         ax.plot(X[:,0],Y_pred[:,0],".",label="pred")
#
#     def generate_XY(self, batch_size) -> tuple:
#         x=tf.random.uniform([batch_size,1])
#         nu_vec=tf.ones([batch_size,1])*self.nu
#         X=tf.concat([x,nu_vec],axis=1)
#
#         Y=tf.sin(x*self.nu)
#         return X,Y
#
#     def loss(self, Y_true, Y_pred) -> tf.Tensor:
#         return tf.reduce_mean(tf.square(Y_true-Y_pred))
#
#     def score(self, model) -> tuple or float:
#         X,Y=self.generate_XY(1000)
#         Y_pred=model(X)
#         return {"dist l1":tf.reduce_mean(tf.abs(Y-Y_pred)).numpy()}
#
#
# def test_numeric_cases():
#     nus_train=[4,8]
#     nus_test=[2,4,6,8,10]
#
#     creator_train={po:DataCreator_num(po) for po in nus_train}
#     creator_test= {po:DataCreator_num(po) for po in nus_test}
#
#     testor=GridUp(
#         SimpleModelClass,
#         creator_train,
#         creator_test,
#         fixed_params={"width":10,"nb_layers":3},
#         #varying_params={"width":[5,10,20],"nb_layers":[1,2,3,4]},
#         varying_params={"width": [5, 10], "nb_layers": [1, 2]},
#         minutes=0.01,
#         verbose=False
#     )
#
#     res=testor.watch_param("width")
#     print(res)
#     testor.plot_last_result(True)
#     # testor.plot_last_result(False)
#     testor.plot_prediction()
#
#     plt.show()
#
#     #
#     # testor.watch_param("nb_layers")
#     # testor.plot_last_result(True)
#     # testor.plot_last_result(False)
#
# #
# # def test_string_cases():
# #
# #     powers_train = ["tf.exp","tf.sin"]
# #     powers_test = ["tf.exp","tf.sin","tf.square","tf.cos"]
# #
# #     creator_train = {po: DataCreator_string(po) for po in powers_train}
# #     creator_test = {po: DataCreator_string(po) for po in powers_test}
# #
# #     testor = GridUp(SimpleModelClass, creator_train, creator_test, fixed_params={"width": 10, "nb_layers": 3},
# #         varying_params={"width": [5, 10, 20], "nb_layers": [1, 2, 3, 4]}, minutes=0.05,verbose=True )
# #
# #     res=testor.watch_param("width")
# #     print(res)
# #     testor.plot_last_result(True)
# #     testor.plot_last_result(False)
# #
# #     testor.watch_param("nb_layers")
# #     testor.plot_last_result(True)
# #     testor.plot_last_result(False)
#
#
# if __name__=="__main__":
#     test_numeric_cases()
#     #test_string_cases()
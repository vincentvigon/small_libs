import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import tensorflow_probability as tfp
import grid_up.grid_up as gr
import fourier_neural_operator.FNO_1d_plus as fno

pp=print



def linspace(a,b,N,dtype):
    if dtype==tf.float32:
        return tf.linspace(a,b,N)
    elif dtype==tf.float64:
        return tf.constant(np.linspace(a,b,N))
    else:
        raise Exception("only for tf.float32 or tf.float64")


class Mesh:
  def __init__(self, N, a, b,dtype):
    self.N = N
    self.a = a
    self.b = b
    self.m = tf.linspace(tf.constant(a,dtype=dtype),tf.constant(b,dtype=dtype),N)
    self.h= tf.abs(self.m[1]-self.m[0])
    self.mid = (self.a + self.b) / 2
    self.L = self.b - self.a


class NewtonData(gr.GridUp_dataMaker):



    def score(self, model) -> dict or float:
        X,Y=self.make_XY(1024)
        Y_pred=model(X)
        residues=self.G2(X,Y_pred)
        return {"residue/N^2":tf.reduce_mean(tf.abs(residues)/self.mesh.N**2)}

    def plot_prediction(self, ax, model: tf.keras.Model) -> None:
        X,Y=self.make_XY(1)
        Y_pred=model(X)
        ax.plot(Y[0,:,0],label="Y_true")
        ax.plot(Y_pred[0,:,0],label="Y_pred")


    def __init__(self,
            a, #deb de l'intervalle
            b, #fin de l'intervalle
            N, #nombre de points
            k, #fonction déterminant l'équation
            kind="fourier", # "fourier" ou "gauss"
            BC="neumann",# "dirichlet" ou "neumann"
            dtype=tf.float32,
            verbose=True):
        self.a=a
        self.b=b
        self.N=N
        self.k=k
        self.kind=kind
        self.BC=BC
        self.dtype = dtype
        self.verbose = verbose

        self.mesh = Mesh(N, self.a, self.b,self.dtype)
        # noinspection PyUnresolvedReferences
        self.dist = tfp.distributions.Normal(loc=tf.constant(0.,dtype=self.dtype), scale=tf.constant(1.,dtype=self.dtype) )
        #self.dist.dtype=self.dtype
        if self.verbose:
            print(f"NewtonData with: nb points={N}, kind of data:{kind}")

        self.normalisation_for_f=1000.


    def elliptic(self,U,alpha):
        # reaction term
        reac = alpha * U
        Dm, Dp=self.first_order_derivative(U)
        diffusion = self.diffusion(Dm,Dp)
        return diffusion+reac


    def diffusion(self,Dm,Dp):
        return (Dm - Dp) / self.mesh.h

    def first_order_derivative(self, U):
        """
                On calcule le second membre de l'EDP:
                    alpha U + nabla(k(U) nabla(U))
                avec par exemple: k(U) = 1 + U^4

                calculé en différence finie,
        """

        h = self.mesh.h  # (N)

        Um = U[:,:-1]  # (N)   Um = U_i-1
        Up = U[:,1:]  # (N)  Up = U_i+1

        if self.BC == "dirichlet":
            Um = tf.concat([U[:, :1 ], Um], axis=1)
            Up = tf.concat( [Up,U[:, -1:]], axis=1)

        elif self.BC == "periodic":
            Um = tf.concat([U[:, -1:], Um], axis=1)
            Up = tf.concat([Up, U[:, :1]], axis=1)
        else:
            raise Exception(f"must be 'dirichlet' or 'periodic': 'neumann' will come soon, found:{self.BC}")

        # diffusion terms
        Kp = 0.5 * (self.k(U) + self.k(Up))
        Km = 0.5 * (self.k(Um) + self.k(U))

        Dp=Kp * (Up - U)/h
        Dm=Km * (U - Um)/h

        return Dm,Dp

    def get_G_for_scipy(self,f,alpha):
        return lambda U:self.G(U,f,alpha)

    def G(self, U,f,alpha):
        """
        Cacule les résidus:
         G(U)= alpha U + nabla(k(U) nabla(U)) - f
        """
        ellip = self.elliptic(U,alpha)
        res = ellip - f
        return res

    def G2(self,X,Y):
        f_nor=X[:,:,0]
        f=f_nor*self.normalisation_for_f
        alpha=X[:,:,1]
        U=Y[:,:,0]
        return self.G(U,f,alpha)

    def generate_fourier(self,nb_data):

        nb_fourier=6
        n = tf.range(1,nb_fourier+1,dtype=self.dtype)[:,None,None]
        an = tf.random.uniform(minval=-1,maxval=1,shape=[nb_fourier,nb_data,1],dtype=self.dtype)
        an/=n #pour que les hautes fréquences soient moins présentes

        nu = 2 * np.pi / self.mesh.L * n

        a0 = tf.random.uniform(minval=-1,maxval=1,shape=[1,nb_data,1],dtype=self.dtype)
        t=self.mesh.m[None,None,:]
        res= a0 + an*tf.sin(t* nu)
        res=tf.reduce_sum(res,axis=0)
        return res

    def generate_gaussian_mix(self,nb_data):

        nb_gauss_max=6

        std = tf.random.uniform(minval=0.025, maxval=0.07,shape=[nb_gauss_max,nb_data,1],dtype=self.dtype)
        mean= tf.random.uniform(minval=self.mesh.mid - 0.25 * self.mesh.L , maxval=self.mesh.mid + 0.25 * self.mesh.L,shape=[nb_gauss_max,nb_data,1],dtype=self.dtype)
        x = (self.mesh.m[None,None,:]-mean)/std

        gauss=self.dist.prob(x)

        mask=tf.random.uniform(minval=0,maxval=1.75,shape=[nb_gauss_max-1,nb_data,1],dtype=self.dtype)
        mask=tf.math.floor(mask) #un peu plus de 0 que de 1 dans ce masque

        mask=tf.concat([tf.ones([1,nb_data,1]),mask],axis=0) #pour avoir au moins une gaussienne à chaque fois
        gauss*=mask
        res=tf.reduce_sum(gauss,axis=0) #on somme les gaussiennes

        res = res / (tf.reduce_max(res,axis=1)[:,None]+0.1) + 0.5

        return res

    def generate_alpha(self,nb_data):
        if self.kind=="gauss":
            return self.generate_gaussian_mix(nb_data)
        elif self.kind=="fourier":
            return self.generate_fourier(nb_data)
        else:
            raise Exception(f"kind must be 'gauss' or 'fourier'. Found:{self.kind}")


    @tf.function
    def make_XY(self,batch_size):
        if self.verbose:
            print("traçage de la fonction generate_XY")
        """"""
        """
        On triche: on génére 
          u,alpha
        On en déduit le second membre
          f 
        qui permet d'avoir un résidu proche de zéro
        
        Mais le réseau de neurone devrai retrouver: 
          Y=[u]
        à partir de
          X=[f,alpha]
        
        
        """
        alpha = self.generate_alpha(batch_size)
        U = self.generate_alpha(batch_size)
        f_nor = self.elliptic(U,alpha)/self.normalisation_for_f
        X = tf.stack([f_nor,alpha],axis=2)
        Y = U[:,:,None]
        return X,Y


    def plot_data(self,X,Y):
        f_nor=X[:, :, 0]
        f = f_nor*self.normalisation_for_f
        alpha=X[:, :, 1]
        U=Y[:, :, 0]
        g=self.G(U,f,alpha)
        nb=5
        fig, ax = plt.subplots(nb, 4, figsize=(16, nb))

        for i in range(nb):
            ax[i,0].plot(self.mesh.m,f_nor[i,:] )
            ax[i,1].plot(self.mesh.m, alpha[i,:])
            ax[i,2].plot(self.mesh.m, U[i,:] )
            ax[i,3].plot(self.mesh.m, g[i,:])
        ax[0, 0].set_title("f_nor")
        ax[0, 1].set_title("alpha")
        ax[0, 2].set_title("U")
        ax[0, 3].set_title("G(U)")
        fig.tight_layout()


    # noinspection PyUnboundLocalVariable
    def losses_fn(self, X, Y, Y_pred, keys):
        U = Y[:, :, 0]
        U_pred = Y_pred[:, :, 0]
        alpha = X[:, :, 1]
        result = {}

        if "D" in keys or "diffusion" in keys or "residues" in keys:
            Dm, Dp = self.first_order_derivative(U)
            Dm_pred, Dp_pred = self.first_order_derivative(U_pred)

        if "diffusion" in keys or "residues" in keys:
            diff = self.diffusion(Dm, Dp)
            diff_pred = self.diffusion(Dm_pred, Dp_pred)

        if "residues" in keys:
            f = X[:, :, 0]*self.normalisation_for_f
            reac_pred = alpha * U_pred
            residues_pred = diff_pred + reac_pred - f

        if "U" in keys:
            diff=U - U_pred
            result["U"] = mse(diff)+mse(alpha*diff)
        if "D" in keys:
            result["D"] = (mse(Dm - Dm_pred) + mse(Dp - Dp_pred))/self.mesh.N
        if "diffusion" in keys:
            result["diffusion"] = mse(diff - diff_pred)/self.mesh.N**2
        if "residues" in keys:
            result["residues"] = mse(residues_pred)/self.mesh.N**2

        return result


def mse(a):
    return tf.reduce_mean(tf.square(a))





def test_losses():
    name_of_losses = ["U", "D", "diffusion", "residues"]

    agent= AgentNewton(
        name_of_losses=name_of_losses,
        modes=20,
        width=20,
        nb_layer=4,
        first_channel_unchanged=True,
        freq_mix_size=5,
        pad_prop=0,
        pad_kind="no_padding",
        batch_size=64,
        only_one_optimizer=True,
        lr=1e-3
    )
    model=agent.get_model()
    data=NewtonData(a=0,b=1,N=100,k=lambda u: u ** 4 + 1.0,kind="gauss",BC="dirichlet")

    X,Y=data.make_XY(100)
    Y=tf.random.uniform(Y.shape)
    Y_pred=tf.random.uniform(Y.shape)

    Dm,Dp=data.first_order_derivative(tf.random.uniform([1,10]))
    print(Dm)
    print(Dp)

    losses=data.losses_fn(X,Y,Y_pred,name_of_losses)
    print("losses",losses)








class AgentNewton(gr.GridUp_agent):

    def __init__(self,
            name_of_losses,
            modes,
            width,
            nb_layer,
            first_channel_unchanged,
            freq_mix_size,
            pad_prop,
            pad_kind,
            batch_size,
            only_one_optimizer,
            lr
    ):

        self.model = fno.FNO1d_plus(modes, width,1,nb_layer,first_channel_unchanged,freq_mix_size,pad_prop,pad_kind)
        self.optimizer = tf.keras.optimizers.Adam()
        self.batch_size = batch_size
        self.only_one_optimizer=only_one_optimizer
        self.name_of_losses=name_of_losses

        if only_one_optimizer:
            self.optimizers=tf.keras.optimizers.Adam(lr)
        else:
            self.optimizers={name:tf.keras.optimizers.Adam(lr) for name in self.name_of_losses}


    def get_model(self):
        return self.model


    @tf.function
    def train_step_with_details(self, data_maker: NewtonData):
        X,Y=data_maker.make_XY(self.batch_size)
        with tf.GradientTape(persistent=True) as tape:
            Y_pred=self.model.call(X)
            losses=data_maker.losses_fn(X,Y,Y_pred,self.name_of_losses)
            loss=sum([loss_val for loss_val in losses.values()])

        tv=self.model.trainable_variables
        if self.only_one_optimizer:
            grad=tape.gradient(loss,tv)
            self.optimizer.apply_gradients(zip(grad,tv))
        else:
            for key in self.name_of_losses:
                grad=tape.gradient(losses[key],tv)
                self.optimizers[key].apply_gradients(zip(grad,tv))

        del tape
        return loss,losses

    def train_step(self, data_maker: NewtonData):
        loss, _=self.train_step_with_details(data_maker)
        return loss



def test_data():
    ku2 = lambda u: u ** 4 + 1.0  # non linéarité
    #kpu2 = lambda u: 4.0 * u * u * u
    newtonData = NewtonData(a=0.0, b=1.0, N=200,BC="dirichlet",kind="fourier",k=ku2,dtype=tf.float64)

    #premier appel
    nb_data=1500

    newtonData.make_XY(nb_data)

    #second appel
    ti0=time.time()
    X_train,Y_train=newtonData.make_XY(nb_data)
    duration=time.time()-ti0
    print(f"duration={duration} for nb_data={nb_data} with nb points={newtonData.mesh.N}")

    print("X_train",X_train.shape,X_train.dtype)
    print("Y_train",Y_train.shape,X_train.dtype)

    newtonData.plot_data(X_train,Y_train)
    plt.show()


def test_agent():
    name_of_losses = ["U", "D", "diffusion", "residues"]

    agent= AgentNewton(
        name_of_losses=name_of_losses,
        modes=20,
        width=20,
        nb_layer=4,
        first_channel_unchanged=True,
        freq_mix_size=5,
        pad_prop=0,
        pad_kind="no_padding",
        batch_size=64,
        only_one_optimizer=True,
        lr=1e-3
    )

    data=NewtonData(a=0,b=1,N=100,k=lambda u: u ** 4 + 1.0,kind="gauss",BC="dirichlet")

    losses_hist = {name: [] for name in name_of_losses}

    for i in range(10):
        loss, losses = agent.train_step_with_details(data)
        for name in name_of_losses:
            losses_hist[name].append(losses[name])
        if i % 10 == 0:
            print(loss)

    print(losses_hist["diffusion"])
    print(name_of_losses)

    for name in name_of_losses:
        style="-"
        if name=="diffusion":
            style="+"
        plt.plot(losses_hist[name],style,label=name)

    plt.yscale("log")
    plt.legend()
    plt.show()




if __name__=="__main__":
    #test_losses()
    test_agent()




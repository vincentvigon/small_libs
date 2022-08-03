import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import tensorflow_probability as tfp

from grid_up.grid_up_old import GridUp_testable

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


class NewtonData(GridUp_testable):  # à modifier /creation des jeux de données

    def loss(self, Y_true, Y_pred) -> tf.Tensor:
        return tf.reduce_mean(tf.square(Y_true-Y_pred))

    def score(self, model) -> dict or float:
        X,Y=self.generate_XY(1024)
        Y_pred=model(X)
        residues=self.G2(X,Y_pred)
        return {"residue/N^2":tf.reduce_mean(tf.abs(residues)/self.mesh.N**2)}

    def plot_prediction(self, ax, model: tf.keras.Model) -> None:
        X,Y=self.generate_XY(1)
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


    def elliptic(self, U,alpha):
        if self.BC == "dirichlet":
            return self.elliptic_dirichlet_periodic(U,alpha,True)
        elif self.BC=="periodic":
            return self.elliptic_dirichlet_periodic(U,alpha,False)
        elif self.BC=="neumann":
            return self.elliptic_neumann(U,alpha)
        else:
            raise Exception(f"must be 'dirichlet', 'neumann' or 'periodic', found:{self.BC}")


    def elliptic_neumann(self, U,alpha):
        """
        alpha U + nabla(k(U) nabla(U))
        ex: k(U) = 1 + U^4
        calculé en différence finie, condition de bord de Neumann
        """

        h = self.mesh.h # (N)

        # Attention: le roll qui peut cacher des erreurs. Je préférerais supprimer les valeurs de bord que les enrouler
        Up = tf.roll(U,-1,axis=1)  #   (N)  Up = U_i+1
        Um = tf.roll(U, 1,axis=1)  #   (N)   Um = U_i-1

        # diffusion terms
        Kp = 0.5 * (self.k(U) + self.k(Up))
        Km = 0.5 * (self.k(Um) + self.k(U))

        diff = - (Kp * (Up - U) - Km * (U - Um)) / h ** 2

        #todo: ici j'ai changé, les conditions de neumann me semblaients bizarre
        diff_0 = - (Kp[:,0] * (U[:,1] - U[:,0])) / h   + U[:,0]   # - Km * (U - Um) = uL neumann on left boundary
        diff_0=diff_0[:,None]
        diff_last = - ( - Km[:,-1] * (U[:,-1] - U[:,-2]) ) / h  + U[:,-1]  # - Kp * (Up - U) = uR neumann on right boundary
        diff_last=diff_last[:,None]

        diff=tf.concat([diff_0,diff[:,1:-1],diff_last],axis=1)

        # reaction term
        reac = alpha * U

        return reac + diff


    def elliptic_dirichlet_periodic(self, U,alpha,is_dirichlet):

        h = self.mesh.h  # (N)

        Um = U[:,:-1]  # (N)   Um = U_i-1
        Up = U[:,1:]  # (N)  Up = U_i+1

        if is_dirichlet:
            Um = tf.concat([U[:, :1 ], Um], axis=1)
            Up = tf.concat( [Up,U[:, -1:]], axis=1)
        else:
            Um = tf.concat([U[:, -1:], Um], axis=1)
            Up = tf.concat([Up, U[:, :1]], axis=1)

        # diffusion terms
        Kp = 0.5 * (self.k(U) + self.k(Up))
        Km = 0.5 * (self.k(Um) + self.k(U))
        diff = - (Kp * (Up - U) - Km * (U - Um)) / h ** 2

        # reaction term
        reac = alpha * U

        return reac+diff


    def elliptic_suggestion(self, U,alpha_loc):
        """
        On pourrait ne mettre aucune condition au limite, en raccoursissant le signal de chaque côté,
        si on ne doit appliquer l'opérateur qu'une seule fois ...
        """
        # reaction term
        h = self.mesh.h # (N)

        Up = U[:,2:]      #  Up = U_i+1
        Um = U[:,:-2]     #  Um = U_i-1
        Uc = U[:,1:-1]    #  Uc = U_i

        # diffusion terms
        Kp = 0.5 * (self.k(Uc) + self.k(Up))
        Km = 0.5 * (self.k(Um) + self.k(Uc))

        diff = - (Kp * (Up - Uc) - Km * (Uc - Um)) / h ** 2

        reac = alpha_loc * Uc

        return reac + diff

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
    def generate_XY(self,batch_size):
        if self.verbose:
            print("traçage de la fonction generate_XY")
        """"""
        """
        On génére 
          u,alpha
        On en déduit le second membre
          f 
        qui permet d'avoir un résidu nul
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


def test():
    ku2 = lambda u: u ** 4 + 1.0  # non linéarité
    #kpu2 = lambda u: 4.0 * u * u * u
    newtonData = NewtonData(a=0.0, b=1.0, N=200,BC="neumann",kind="fourier",k=ku2,dtype=tf.float64)

    #premier appel
    nb_data=1500

    newtonData.generate_XY(nb_data)

    #second appel
    ti0=time.time()
    X_train,Y_train=newtonData.generate_XY(nb_data)
    duration=time.time()-ti0
    print(f"duration={duration} for nb_data={nb_data} with nb points={newtonData.mesh.N}")

    print("X_train",X_train.shape,X_train.dtype)
    print("Y_train",Y_train.shape,X_train.dtype)

    newtonData.plot_data(X_train,Y_train)
    plt.show()


def test_generate_fourier():

    def one():
        nb_data=1

        nb_fourier=6
        dtype=tf.float32

        n = tf.range(1,nb_fourier+1,dtype=dtype)[:,None,None]

        pp("n",n)

        an = tf.random.uniform(minval=-1,maxval=1,shape=[nb_fourier,nb_data,1],dtype=dtype)
        pp("an",an)
        an/=n #pour que les hautes fréquences soient moins présentes
        pp("an",an)

        mesh=Mesh(200,0,1,dtype)

        nu = 2 * np.pi / mesh.L * n

        a0 = tf.random.uniform(minval=-1,maxval=1,shape=[1,nb_data,1],dtype=dtype)

        t=mesh.m[None,None,:]
        res= a0 + an*tf.sin(t* nu)
        res=tf.reduce_sum(res,axis=0)
        return tf.reduce_sum(res)

    OK=True
    while OK:
        res=one().numpy()
        if np.isnan(res):
            break









if __name__=="__main__":
    test_generate_fourier()
    #test()




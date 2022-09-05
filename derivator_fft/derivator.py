
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from derivator_fft.fft_nd import Fft_nd
from paddings.paddings import pad_nd, unpad_nd, smooth_padding

pp=print


class Derivator_fft:

    def __init__(self,  axes, interval_lenghts, formula, pad_prop=0.1, suppress_padding_on_output=True,graph_acceleration=False,verbose=False):
        self.axes=axes
        self.interval_lenghts=interval_lenghts
        self.formula=formula
        self.pad_prop=pad_prop
        self.suppress_padding_on_output=suppress_padding_on_output
        self.graph_acceleration=graph_acceleration
        self.verbose=verbose

        self.init_was_made=False


    def init(self, U_example):

        input_shape=U_example.shape
        U_example=tf.constant(U_example) # au cas où les tenseurs sera de type numpy
        self.initial_dtype=U_example.dtype

        sizes = [input_shape[i] for i in self.axes]

        if self.initial_dtype==tf.float32 or self.initial_dtype==tf.complex64:
            is_32bit=True
        elif self.initial_dtype ==tf.float64 or self.initial_dtype ==tf.complex128:
            is_32bit=False
        else:
            raise Exception("the tensor given must have dtype in [tf.float32,tf.complex64,tf.float64,tf.complex128]")


        if is_32bit:
            self.dtype_real = np.float32  # tensorflow accepte les dtype numpy
            self.dtype_complex = np.complex64
        else:
            self.dtype_real = np.float64
            self.dtype_complex = np.complex128


        sizes_with_pad = []
        interval_length_with_pad = []

        pads = [int(size * self.pad_prop) for size in sizes]


        for i in range(len(self.axes)):
            if pads[i] >= sizes[i]:
                pads[i] = sizes[i] - 1
            sizes_with_pad.append(sizes[i] + 2 * pads[i])
            interval_length_with_pad.append(self.interval_lenghts[i] * (1 + 2 * self.pad_prop))

        self.pads = pads

        ks = []

        """
        Calcul des vecteurs k, qui, une fois multiplié à la transformée de fourier, produirons la dérivation 
        """
        for i in range(len(self.axes)):
            k_max = sizes_with_pad[i] // 2
            if sizes_with_pad[i] % 2 == 0:
                k = tf.concat([tf.range(0., k_max, dtype=self.dtype_real), tf.range(-k_max, 0., dtype=self.dtype_real)],
                    axis=0)
            else:
                k = tf.concat(
                    [tf.range(0., k_max, dtype=self.dtype_real), [0.], tf.range(-k_max, 0., dtype=self.dtype_real)],
                    axis=0)

            k *= 2. * np.pi / interval_length_with_pad[i]
            # À cause de la multiplication par sqrt(-1), k passe en second argument ci-dessous
            k = tf.complex(tf.zeros_like(k),k)

            """
            Maintenant, il faut étaler ce vecteur dans le bon axe. Par exemple si current_axis=3
            la shape de k sera (1,1,1,size,1,1,...)
            """
            s = [1] * self.axes[i]
            s += [sizes_with_pad[i]]
            s += [1] * (len(input_shape) - self.axes[i]-1)

            k = tf.reshape(k, s)
            if self.verbose:
                print(i,"shape ieme tenseur:",s)
            ks.append(k)

        self.freq_factor=self.formula(*ks)

        self.result_type = None
        if isinstance(self.freq_factor,dict):
            self.user_keys=self.freq_factor.keys()
            self.freq_factor=self.freq_factor.values()
            self.result_type = "dict"
        elif isinstance(self.freq_factor,list):
            self.result_type="list"
        elif isinstance(self.freq_factor, tuple):
            self.result_type = "tuple"
        else:
            self.result_type = "scalar"
            self.freq_factor =[self.freq_factor]


        self.fft=Fft_nd(self.axes)

        self.ifft=Fft_nd(self.axes,inverse=True)


    def __call__(self,U):

        if self.init_was_made:
            self.init(U)
            self.init_was_made=True

        if self.graph_acceleration:
            print("traçage de la méthode de dérivation du Derivator_fft",end="")
            ti0=time.time()
            tf_function_obj = tf.function(self.D_initial)
            concrete_function = tf_function_obj.get_concrete_function(U)
            duration=time.time()-ti0
            print(", temps de traçage:",duration)
            return concrete_function(U)
        else:
            return self.D_initial(U)


    #
    # @tf.function
    # def D_graph(self,U):
    #     print("traçage de la méthode D_graph de la classe Derivator_fft")
    #     return self.D_initial(U)


    def D_initial(self, U):
        if isinstance(U, np.ndarray):
            U = tf.constant(U)

        U = pad_nd(U, "smooth_periodizing_padding", self.pads, axes=self.axes)
        U = tf.cast(U,self.dtype_complex)


        U_ = self.fft(U)

        res=[]
        for freq_factor in self.freq_factor:
            U_der=U_*freq_factor
            U_der=self.ifft(U_der)

            if self.suppress_padding_on_output:
                U_der=unpad_nd(U_der,self.pads,self.axes)

            U_der = tf.cast(U_der, self.initial_dtype)
            res.append(U_der)


        if self.result_type=="dict":
            res_dict={}
            for key,val in zip(self.user_keys,res):
                res_dict[key]=val
            res=res_dict
        elif self.result_type=="scalar":
            res=res[0]
        elif self.result_type=="list":
            pass
        elif self.result_type=="tuple":
            res=tuple(res)
        return res



def Dx(W,h,keep_size=True,centred=True,axis=-1):

    permut=None
    if axis!=-1:
        dim=len(W.shape)
        permut = np.arange(dim)
        permut[-1] = axis
        permut[axis] = dim - 1
        W=tf.transpose(W,permut)

    if centred:
        Wx=(W[...,2:]-W[...,:-2])/(2*h)
        if keep_size:
            Wx=smooth_padding(Wx,1)
    else:
        Wx=(W[...,1:]-W[...,:-1])/h
        if keep_size:
            Wx=smooth_padding(Wx,[1,0])

    if axis != -1:
        Wx = tf.transpose(Wx, permut)

    return Wx



def test_1d():

    L=3.4
    # si freq est un entier: pas besoin de mettre du padding
    freq=5.15
    cst=freq*2*np.pi/L
    fn=lambda x:tf.sin(cst*x**2)
    fn_x=lambda x: cst*tf.cos(cst*x**2)*2*x

    N=200
    x = np.linspace(0., L, N, endpoint=False).astype(np.float32)
    F = fn(x)
    F_x = fn_x(x)

    F = F[None, None, :, None]
    F_x = F_x[None, None, :, None]

    derivator = Derivator_fft( [2], [L], lambda x: x, pad_prop=0.2)
    F_x_pred = derivator(F)

    fig,axs=plt.subplots(3,1,sharex="all")
    axs[0].set_title("function")
    axs[0].plot(F[0,0,:,0])
    axs[1].set_title("true derivative")
    axs[1].plot(F_x[0,0,:,0])
    axs[2].set_title("pred derivative")
    axs[2].plot(F_x_pred[0,0,:,0])

    fig.tight_layout()
    plt.show()


def test_precision():

    L=3.4
    # si freq est un entier: pas besoin de mettre du padding
    freq=5.15
    cst=freq*2*np.pi/L
    fn=lambda x:tf.sin(cst*x**2)
    fn_x=lambda x: cst*tf.cos(cst*x**2)*2*x

    fig,axs=plt.subplots(4,1,figsize=(10,10))

    Ns = [15, 30, 50, 75, 100, 200, 300, 400, 800,1500]

    errors_df_1=[]
    errors_df_2=[]

    for N in Ns:
        x = np.linspace(0., L, N, endpoint=False).astype(np.float32)
        F = fn(x)
        F_x = fn_x(x)
        F_x_df = Dx(F, L / N)
        error_df_1 = tf.reduce_mean(tf.abs(F_x_df - F_x))
        error_df_2 = tf.reduce_mean(tf.square(F_x_df - F_x))

        errors_df_1.append(error_df_1)
        errors_df_2.append(error_df_2)


    def one_pad_prop(pad_prop):
        print("pad_prop",pad_prop)

        def one_N(N:int):
            print(f"test with N={N}")

            x=np.linspace(0.,L,N,endpoint=False).astype(np.float32)
            F=fn(x)
            F_x=fn_x(x)

            F=F[None,None,:,None]
            F_x=F_x[None,None,:,None]

            derivator=Derivator_fft([2],[L],lambda x:x,pad_prop)#FftDerivator1d(L,F_x.shape,axis=2,pad_prop=pad_prop)
            F_x_pred=derivator(F)
            error_1=tf.reduce_mean(tf.abs(F_x_pred-F_x))
            error_2=tf.reduce_mean(tf.square(F_x_pred-F_x))

            return error_1,error_2

        errors_1=[]
        errors_2=[]

        for N in Ns:
            error_1,error_2=one_N(N)
            errors_1.append(error_1)
            errors_2.append(error_2)

        axs[0].plot(Ns,errors_1,"o-",label=f"mae, pad_prop={pad_prop}")
        axs[1].plot(Ns,errors_2,"o-",label=f"mse, pad_prop={pad_prop}")


    for pad_prop in [0,0.1,0.2,0.5,1]:
        one_pad_prop(pad_prop)

    axs[0].plot(Ns,errors_df_1,"o-",label="mae, finite diff")
    axs[1].plot(Ns,errors_df_2,"o-",label="mse, finite diff")

    axs[0].set_yscale("log")
    axs[1].set_yscale("log")

    axs[0].legend()

    t=tf.linspace(0.,L,1000)
    F=fn(t)
    Fx=fn_x(t)
    axs[2].plot(t,F,label='fonc')
    axs[3].plot(t,Fx,label='der')
    axs[2].legend()
    axs[3].legend()

    plt.tight_layout()
    plt.show()


def simple_test():
    T0=1
    T1=2
    T2=1.5
    a0=tf.linspace(0.,T0,50)[:,None,None]
    a1=tf.linspace(0.,T1,30)[None,:,None]
    a2=tf.linspace(0.,T2,20)[None,None,:]

    U=a0*a1**2*a2
    print("U",U.shape)

    derivator=Derivator_fft([2,1],[T0,T1],lambda a0,a1:[a0,a1],pad_prop=0,suppress_padding_on_output=False)
    U_a0,U_a1=derivator(U)

    print(U_a0.shape)

    fig,axs=plt.subplots(2,1)
    axs[0].imshow(U_a0[:,:,0])
    axs[1].imshow(U_a1[:,:,0])
    plt.show()



def test_divergence():
    N=10
    a0=tf.linspace(0,1,N)
    a1=tf.linspace(0,1,N)
    a2=tf.linspace(0,1,N)
    a3=tf.linspace(0,1,N)

    a0_=a0[:,None,None,None]
    a1_=a1[None,:,None,None]
    a2_=a2[None,None,:,None]
    a3_=a3[None,None,None,:]

    func=lambda a0,a1,a2,a3:a0**2+a1**2+a2**2+a3**2
    div_func=lambda a0,a1,a2,a3: 2*a0+2*a1+2*a2+2*a3

    F=func(a0_,a1_,a2_,a3_)
    div_F=div_func(a0_,a1_,a2_,a3_)

    derivator_fft=Derivator_fft([0,1,2,3],[1,1,1,1],lambda a0,a1,a2,a3:a0+a1+a2+a3,graph_acceleration=True)

    for i in range(4):
        ti0=time.time()
        div_F_pred=derivator_fft(F)
        duration=time.time()-ti0
        print(f"temps d'execution {i}:",duration)


    print("SANS TRAçAGE")
    derivator_fft=Derivator_fft([0,1,2,3],[1,1,1,1],lambda a0,a1,a2,a3:a0+a1+a2+a3,graph_acceleration=False)

    for i in range(4):
        ti0=time.time()
        div_F_pred=derivator_fft(F)
        duration=time.time()-ti0
        print(f"temps d'execution {i}:",duration)


def test_2d():

    # si freq est un entier: pas besoin de mettre du padding
    freq=0.3
    cst=freq*2*np.pi
    fn=lambda a0,a1:tf.sin(cst*a0**2)*a1
    fn_a0=lambda a0,a1: tf.cos(cst*a0**2)*(cst*2*a0)*a1
    fn_a1=lambda a0,a1:tf.sin(cst*a0**2)

    fn_a0a0 = lambda a0, a1: tf.cos(cst*a0**2)*(cst*2*a0)**2*a1+tf.sin(cst*a0**2)*(cst*2*a0)*a1  \
                             + tf.cos(cst*a0**2)*(cst*2)*a1
    fn_a1a1 = lambda a0, a1: tf.zeros_like(a1)
    fn_a0a1=lambda a0,a1: tf.cos(cst*a0**2)*(cst*2*a0)

    Ns=[200,300]
    Ls=[2.4,3.4]

    a0 = np.linspace(0., Ls[0], Ns[0]).astype(np.float32)
    a1 = np.linspace(0., Ls[1], Ns[1]).astype(np.float32)
    # Attention, meshgrid est prévu pour des absicisse, ordonnée. Il faut donc l'inverser
    aa0,aa1=tf.meshgrid(a0,a1)
    aa0, aa1=tf.transpose(aa0),tf.transpose(aa1)
    print("aa",aa0.shape,aa1.shape)

    F = fn(aa0,aa1)
    DF={}
    for f,name in zip([fn_a0,fn_a1,fn_a0a0,fn_a0a1,fn_a1a1],["a0","a1","a0^2","a0a1","a1^2"]):
        DF[name]=f(aa0,aa1)

    derivator = Derivator_fft([0, 1], Ls,
        formula=lambda a0, a1: {"a0": a0, "a1": a1, "a0^2": a0 ** 2, "a1^2": a1 ** 2,"a0a1":a0*a1})
    DF_fft=derivator(F)

    extent = [0., Ls[1], 0., Ls[0]]
    fig, axs = plt.subplots(6,2, figsize=(5, 10),sharex="all")
    axs[0, 0].imshow(F, extent=extent, cmap="jet", origin="lower")
    axs[0, 0].set_title("initial function")

    vmin,vmax=-4,4
    for i,key in enumerate(["a0","a1","a0^2","a0a1","a1^2"]):
        axs[i+1,0].imshow(DF[key],extent=extent,cmap="jet",origin="lower",vmin=vmin,vmax=vmax)
        axs[i + 1, 0].set_title(key+"_true")
        axs[i+1,1].imshow(DF_fft[key],extent=extent,cmap="jet",origin="lower",vmin=vmin,vmax=vmax)
        axs[i + 1, 1].set_title(key+"_pred")

    plt.tight_layout()
    plt.show()





if __name__=="__main__":
    #test_1d()
    #test_2d()
    #simple_test()
    #test_precision()
    test_divergence()








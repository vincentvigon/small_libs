
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from paddings.paddings import smooth_periodizing_padding, smooth_padding

pp=print
"""
Signal have shape (batch_size,N,...)
They are discretization of a periodic signal defined on [0,L]
"""


class FftDerivator1d:

    def __init__(self,interval_lenght,input_shape,axis,is_32bit=True,pad_prop=0.1,suppress_padding_on_output=True):

        nb_points = input_shape[axis]
        self.axis=axis

        if is_32bit:
            self.dtype_real=tf.float32
            self.dtype_complex=tf.complex64

        else:
            self.dtype_real = tf.float64
            self.dtype_complex = tf.complex128

        self.suppress_padding_on_output=suppress_padding_on_output

        self.dim=len(input_shape)
        assert self.axis<self.dim


        self.pad = int(nb_points * pad_prop)
        if self.pad >= nb_points:
            self.pad = nb_points - 1

        if self.pad == 0 : #todo à quoi ça sert ?
            self.interval_length_with_pad=interval_lenght
            self.nb_points_with_pad=nb_points

        self.nb_points_with_pad  = nb_points +2*self.pad
        self.interval_length_with_pad  = interval_lenght*(1+2*pad_prop)

        print(f"le signal sera prolongé de {self.pad} de chaque côté")

        if self.nb_points_with_pad%2 ==0:
            k_max = self.nb_points_with_pad//2
            k_x = tf.concat([tf.range(0., k_max,dtype=self.dtype_real), tf.range(-k_max, 0.,dtype=self.dtype_real)], axis=0)
        else:
            k_max = self.nb_points_with_pad // 2
            k_x = tf.concat([tf.range(0., k_max,dtype=self.dtype_real),[0.],tf.range(-k_max, 0.,dtype=self.dtype_real)], axis=0)

        k_x=k_x*tf.cast(2.*np.pi/self.interval_length_with_pad,self.dtype_real)
        #multiplier par i inverse la partie réelle et imaginaire
        k_x = tf.complex(tf.zeros_like(k_x,dtype=self.dtype_real),k_x)

        shape=np.ones(self.dim,dtype=np.int32)
        shape[-1]=self.nb_points_with_pad

        self.k_x=tf.reshape(k_x,shape)

        self.permut=np.arange(self.dim)
        self.permut[-1]=axis
        self.permut[axis]=self.dim-1


    def Dx(self,U):
        if isinstance(U,np.ndarray):
            U=tf.constant(U)

        initial_dtype=U.dtype

        if initial_dtype==tf.float32:
            U=tf.cast(U,tf.complex64)
        elif initial_dtype==tf.float64:
            U=tf.cast(U,tf.complex128)

        if self.axis!=self.dim-1:
            U=tf.transpose(U,self.permut)

        if self.pad>0:
            U=smooth_periodizing_padding(U,self.pad)#les autres padding fonctionnent vraiment moins bien

        U_  =tf.signal.fft(U)
        U_  *=self.k_x
        U_x =tf.signal.ifft(U_)

        if self.pad>0 and self.suppress_padding_on_output:
            U_x=U_x[...,self.pad:-self.pad]

        if self.axis!=self.dim-1:
            U_x=tf.transpose(U_x,self.permut)

        U_x=tf.cast(U_x,initial_dtype)

        return U_x


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

def test_repeat():
    N=6
    k_max=3
    k_x = tf.concat([tf.range(0., k_max), tf.range(-k_max, 0.)],axis=0)
    k_x = k_x[ :,None]
    k_x = tf.repeat(k_x, N, axis=1)
    print("k_x",k_x)

    k_y = tf.concat([tf.range(0., k_max), tf.range(-k_max, 0.)],axis=0)
    k_y = k_y[None, :]
    k_y = tf.repeat(k_y, N, axis=0)
    print("k_y",k_y)

def test_best_pad_prop():

    L=1.5
    # si freq est un entier: pas besoin de mettre du padding
    freq=3.7
    cst=freq*2*np.pi/L
    fn=lambda x:tf.sin(cst*x)
    fn_x=lambda x: cst*tf.cos(cst*x)

    def one(N:int):
        print(f"test with N={N}")

        x=np.linspace(0.,L,N,endpoint=False).astype(np.float32)
        F=fn(x)
        F_x=fn_x(x)
        F=F[None,None,:,None]
        F_x=F_x[None,None,:,None]

        errors=[]
        durations=[]
        pad_props=tf.linspace(0,1,20)
        for pad_prop in pad_props:
            ti0=time.time()
            derivator=FftDerivator1d(L,F_x.shape,axis=2,pad_prop=pad_prop,suppress_padding_on_output=True)
            F_x_pred=derivator.Dx(F)
            error=tf.reduce_mean(tf.abs(F_x_pred-F_x))
            errors.append(error)
            durations.append(time.time()-ti0)

        return errors,durations,pad_props

    Ns=[30,50,100,200,400]
    fig,axs=plt.subplots(len(Ns),1,figsize=(10,len(Ns)*1.5))
    for i,N in enumerate(Ns):
        errors, durations,pad_props=one(N)
        axs[i].plot(pad_props,errors)
        axs[i].set_xlabel("pad proportion")
        axs[i].set_ylabel("error")
        axs[i].set_title("nb points="+str(N))

    plt.tight_layout()
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

            derivator=FftDerivator1d(L,F_x.shape,axis=2,pad_prop=pad_prop)
            F_x_pred=derivator.Dx(F)
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

def test_padding():

    L=1.5
    # si freq est un entier: pas besoin de mettre du padding
    freq=3.7
    cst=freq*2*np.pi/L
    fn=lambda x:tf.sin(cst*x)
    fn_x=lambda x: cst*tf.cos(cst*x)

    def one(N:int,is_32bit:bool):
        print(f"test with N={N}, is_32bit={is_32bit}")
        if is_32bit:
            dtype_np=np.float32
        else:
            dtype_np=np.float64
        x=np.linspace(0.,L,N,endpoint=False).astype(dtype_np)
        F=fn(x)
        F_x=fn_x(x)
        F=F[None,None,:,None]
        F_x=F_x[None,None,:,None]

        fig,axs=plt.subplots(4,1,figsize=(10,10))
        axs[0].plot(x,F[0,0,:,0],label="F")
        pp(F.shape)
        F_x_fd=Dx(F,L/N,axis=2)
        F_x_fdn=Dx(F,L/N,axis=2,centred=False)
        pp(F_x_fd.shape)

        axs[1].plot(x,F_x[0,0,:,0],label="F_x true")
        axs[1].plot(x,F_x_fd[0,0,:,0],label="finite diff centred")
        axs[1].plot(x,F_x_fdn[0,0,:,0],label="finite diff non-centred")

        for pad_prop in [0.,0.2,0.5,0.9]:
            derivator=FftDerivator1d(L,F_x.shape,axis=2,is_32bit=is_32bit,pad_prop=pad_prop,suppress_padding_on_output=True)
            F_x_pred=derivator.Dx(F)
            axs[2].plot(x,F_x_pred[0,0,:,0],label=f"pad_prop:{pad_prop}")
            axs[3].plot(x[:30],F_x_pred[0,0,:30,0],label=f"pad_prop:{pad_prop}")

        axs[0].legend()
        axs[1].legend()
        axs[2].legend()

        plt.show()

    N=101
    one(N,True)


if __name__=="__main__":
    #test_repeat()
    #test_padding()
    #test_periodizing()
    test_precision()
    #test_best_pad_prop()

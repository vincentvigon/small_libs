
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from paddings.paddings import smooth_periodizing_padding, smooth_padding, pad_nd,kind_dict

pp=print
"""
Signal have shape (batch_size,N,...)
They are discretization of a periodic signal defined on [0,L]
"""


#
# class T3FTDerivator:
#
#     def __init__(self, U_example,  axes, interval_lenghts, formula, pad_prop=0.1, suppress_padding_on_output=True):
#
#         if len(axes)%3==0:
#             nb_derivator=len(axes)//3
#         else:
#             nb_derivator = len(axes) // 3 + 1
#
#         self.derivators=[]
#         for i in range(nb_derivator):
#             loc_var_1=""
#             loc_v=""
#             for j in range(len(axes)):
#                 if 3*i<=j<3*(i+1):
#                     loc_var_1+="x"+str(j%3)+","
#                     loc_v+="x"+str(j%3)+","
#                 else:
#                     loc_var_1+="1,"
#             loc_var_1=loc_var_1[:-1] #suppression de la dernière virgule
#             loc_v=loc_v[:-1]
#             loc_fun_str="lambda "+loc_v+": formula("+loc_var_1+")"
#             loc_formula=eval(loc_fun_str)
#             loc_axes=axes[3*i:3*(i+1)]
#             loc_interval_lengths=interval_lenghts[3*i:3*(i+1)]
#
#             self.derivators.append(_T3FTDerivator_123d(U_example,  loc_axes, loc_interval_lengths, loc_formula, pad_prop, suppress_padding_on_output))
#
#     def D(self,U):
#
#         for derivator in self.derivators:
#             U=derivator.D(U)
#
#         return U

"""
sur GPU l'utilisation de fft2d est significativement plus rapide que l'utilisation de deux fois fft1d
"""
def test_fft_perf():
    print("traçage")

    a, b, c, d = 10, 2000, 1000, 10
    A = tf.random.uniform([a, b, c, d])
    A = tf.complex(A, tf.zeros_like(A))

    def complex_one(shape):
        res = tf.ones(shape)
        res = tf.complex(res, tf.zeros_like(res))
        return res

    mult1 = complex_one([1, b, 1, 1])
    mult2 = complex_one([1, 1, c, 1])
    mult12 = complex_one([1, b, c, 1])

    def fft_one_ax(X, axis):
        dim = len(X.shape)
        permu = np.arange(4)
        permu[-1] = axis
        permu[axis] = dim - 1
        X = tf.transpose(X, permu)
        X = tf.signal.fft(X) * complex_one([1, 1, 1, X.shape[3]])
        X = tf.transpose(X, permu)
        return X

    def fft_two_ax(X, axes):
        dim = len(X.shape)

        permu = np.arange(4)
        permu[-1] = axes[0]
        permu[axes[0]] = dim - 1
        X = tf.transpose(X, permu)

        permu_ = np.arange(4)
        permu_[-1] = axes[1]
        permu_[axes[1]] = dim - 1
        X = tf.transpose(X, permu_)

        X = tf.signal.fft2d(X) * complex_one([1, 1, X.shape[2], X.shape[3]])
        X = tf.transpose(X, permu_)
        X = tf.transpose(X, permu)
        return X

    ti0 = time.time()
    X = fft_one_ax(A, 1)
    X_1 = fft_one_ax(X, 2)
    duration = time.time() - ti0
    tf.print("duration un ax", duration)

    ti0 = time.time()
    X_2 = fft_two_ax(A, [1, 2])
    duration = time.time() - ti0
    tf.print("duration two ax", duration)

    diff = tf.reduce_sum(tf.abs(X_1 - X_2))
    print("diff", diff)
    return diff



def create_permut(axes,dim):
    last_index=np.arange(dim-len(axes),dim)
    to_move=[]
    place=[]
    for i in axes:
        if i<dim-len(axes):
            to_move.append(i)
    for j in last_index:
        if j not in axes:
            place.append(j)
    res=np.arange(dim)
    for i,j in zip(to_move,place):
        res[i]=j
        res[j]=i
    return res


def test_create_permut():
    res=create_permut([0,2],4)
    print(res)







def test_T3FT():
    # dim=4
    # U_example=tf.ones(np.arange(dim)+3)
    # axes=np.arange(dim)
    # interval_lenghts=np.arange(dim,dtype=np.float32)*10
    # var=""
    # for i in range(dim):
    #     var+="a"+str(i)+","
    # var=var[:-1]
    # formula_str="lambda "+var+": tf.reduce_sum(["+var+"])"
    # print(formula_str)
    #
    # formula=eval(formula_str)
    # derivator=_T3FTDerivator_123d(U_example,  axes, interval_lenghts, formula)

    U_example = tf.ones([4,5,6,7,8,9])
    axes=[1,2,3,4]
    interval_lenghts=[10,20,30,40]
    formula=lambda x,y,z,t:x+y+z+t
    derivator = _T3FTDerivator_123d(U_example, axes, interval_lenghts, formula)


#
# class _T3FTDerivator_123d:
#
#     def __init__(self, U_example,  axes, interval_lenghts, formula, pad_prop=0.1, suppress_padding_on_output=True):
#
#         self.formula=formula
#         self.dim_deriv = len(axes)
#
#         input_shape=U_example.shape
#         U_example=tf.constant(U_example) # au cas où les tenseurs sera de type numpy
#         self.initial_dtype=U_example.dtype
#
#         self.axes_after_transpose = np.arange(len(input_shape)-self.dim_deriv,len(input_shape))#tuple(range(-self.dim_deriv, 0))  # ex: (-3,-2,-1)
#
#         self.suppress_padding_on_output = suppress_padding_on_output
#
#
#         sizes = [input_shape[i] for i in axes]
#
#         if self.initial_dtype==tf.float32 or self.initial_dtype==tf.complex64:
#             is_32bit=True
#         elif self.initial_dtype ==tf.float64 or self.initial_dtype ==tf.complex128:
#             is_32bit=False
#         else:
#             raise Exception("the tensor given must have dtype in [tf.float32,tf.complex64,tf.float64,tf.complex128]")
#
#         if is_32bit:
#             self.dtype_real = np.float32  # tensorflow accepte les dtype numpy
#             self.dtype_complex = np.complex64
#         else:
#             self.dtype_real = np.float64
#             self.dtype_complex = np.complex128
#
#
#         pads = [int(size * pad_prop) for size in sizes]
#         sizes_with_pad = []
#         interval_length_with_pad = []
#
#         for i in range(self.dim_deriv):
#             if pads[i] >= sizes[i]:
#                 pads[i] = sizes[i] - 1
#             sizes_with_pad.append(sizes[i] + 2 * pads[i])
#             interval_length_with_pad.append(interval_lenghts[i] * (1 + 2 * pad_prop))
#         self.pads = pads
#
#         ks = []
#         for i in range(self.dim_deriv):
#             k_max = sizes_with_pad[i] // 2
#             if sizes_with_pad[i] % 2 == 0:
#                 k = tf.concat([tf.range(0., k_max, dtype=self.dtype_real), tf.range(-k_max, 0., dtype=self.dtype_real)],
#                     axis=0)
#             else:
#                 k = tf.concat(
#                     [tf.range(0., k_max, dtype=self.dtype_real), [0.], tf.range(-k_max, 0., dtype=self.dtype_real)],
#                     axis=0)
#
#             k *= 2. * np.pi / interval_length_with_pad[i]
#             # À cause de la multiplication par sqrt(-1), k passe en second argument ci-dessous
#             k = tf.complex(tf.zeros_like(k),k)
#
#             s = [1] * (len(input_shape) - self.dim_deriv + i)
#             s += [sizes_with_pad[i]]
#             s += [1] * (self.dim_deriv - 1 - i)
#
#             k = tf.reshape(k, s)
#             pp(i,"shape ieme tenseur:",s)
#             ks.append(k)
#
#         self.freq_factor=formula(*ks)
#         self.result_type = None
#
#         if isinstance(self.freq_factor,dict):
#             self.user_keys=self.freq_factor.keys()
#             self.freq_factor=self.freq_factor.values()
#             self.result_type = "dict"
#         elif isinstance(self.freq_factor,list):
#             self.result_type="list"
#         elif isinstance(self.freq_factor, tuple):
#             self.result_type = "tuple"
#         else:
#             self.result_type = "scalar"
#             self.freq_factor =[self.freq_factor]
#
#         self.permut = None
#         if self.dim_deriv == 1:
#             self.permut = np.arange(len(input_shape))
#             self.permut[-1] = axes[0]
#             self.permut[axes[0]] = len(input_shape) - 1
#         elif self.dim_deriv == 2:
#             self.permut = np.arange(len(input_shape))
#             self.permut[-2] = axes[0]
#             self.permut[-1] = axes[1]
#             self.permut[axes[0]] = len(input_shape) - 2
#             self.permut[axes[1]] = len(input_shape) - 1
#         elif self.dim_deriv == 3:
#             self.permut = np.arange(len(input_shape))
#             self.permut[-3] = axes[0]
#             self.permut[-2] = axes[1]
#             self.permut[-1] = axes[2]
#
#             self.permut[axes[0]] = len(input_shape) - 3
#             self.permut[axes[1]] = len(input_shape) - 2
#             self.permut[axes[2]] = len(input_shape) - 1
#
#         if tuple(self.permut)==tuple(range(len(input_shape))):
#             print("pas besoin de permuter")
#             self.permut=None
#
#
#     def fft(self, U):
#         if self.dim_deriv == 1:
#             return tf.signal.fft(U)
#         elif self.dim_deriv == 2:
#             return tf.signal.fft2d(U)
#         elif self.dim_deriv == 3:
#             return tf.signal.fft3d(U)
#         else:
#             raise Exception("dim must be <=4")
#
#     def ifft(self, U):
#         if self.dim_deriv == 1:
#             return tf.signal.ifft(U)
#         elif self.dim_deriv == 2:
#             return tf.signal.ifft2d(U)
#         elif self.dim_deriv == 3:
#             return tf.signal.ifft3d(U)
#         else:
#             raise Exception("dim must be <=4")
#
#     def D(self, U):
#         if isinstance(U, np.ndarray):
#             U = tf.constant(U)
#
#         if self.permut is not None:
#             U = tf.transpose(U, self.permut)
#
#         U = pad_nd(U, "smooth_periodizing_padding", self.pads, axes=self.axes_after_transpose)
#         U = tf.cast(U,self.dtype_complex)
#         U_ = self.fft(U)
#
#
#         res=[]
#         for freq_factor in self.freq_factor:
#             U_der=U_*freq_factor
#             U_der=self.ifft(U_der)
#
#             if self.suppress_padding_on_output:
#                 if len(self.pads) == 1:
#                     U_der = U_der[..., self.pads[0]:-self.pads[0]]
#                 elif len(self.pads) == 2:
#                     U_der = U_der[..., self.pads[0]:-self.pads[0], self.pads[1]:-self.pads[1]]
#                 elif len(self.pads) == 3:
#                     U_der = U_der[..., self.pads[0]:-self.pads[0], self.pads[1]:-self.pads[1], self.pads[2]:-self.pads[2]]
#
#             if self.permut is not None:
#                 U_der = tf.transpose(U_der, self.permut)
#
#             U_der = tf.cast(U_der, self.initial_dtype)
#             res.append(U_der)
#
#         if self.result_type=="dict":
#             res_dict={}
#             for key,val in zip(self.user_keys,res):
#                 res_dict[key]=val
#             res=res_dict
#         elif self.result_type=="scalar":
#             res=res[0]
#         elif self.result_type=="list":
#             pass
#         elif self.result_type=="tuple":
#             res=tuple(res)
#
#         return res



class _T3FTDerivator_123d:

    def __init__(self, U_example,  axes, interval_lenghts, formula, pad_prop=0.1, suppress_padding_on_output=True):

        self.formula=formula
        self.dim_deriv = len(axes)

        input_shape=U_example.shape
        U_example=tf.constant(U_example) # au cas où les tenseurs sera de type numpy
        self.initial_dtype=U_example.dtype

        self.axes_after_transpose = np.arange(len(input_shape)-self.dim_deriv,len(input_shape))#tuple(range(-self.dim_deriv, 0))  # ex: (-3,-2,-1)

        self.suppress_padding_on_output = suppress_padding_on_output

        sizes = [input_shape[i] for i in axes]

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


        pads = [int(size * pad_prop) for size in sizes]
        sizes_with_pad = []
        interval_length_with_pad = []

        for i in range(self.dim_deriv):
            if pads[i] >= sizes[i]:
                pads[i] = sizes[i] - 1
            sizes_with_pad.append(sizes[i] + 2 * pads[i])
            interval_length_with_pad.append(interval_lenghts[i] * (1 + 2 * pad_prop))
        self.pads = pads

        ks = []
        for i in range(self.dim_deriv):
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

            s = [1] * (len(input_shape) - self.dim_deriv + i)
            s += [sizes_with_pad[i]]
            s += [1] * (self.dim_deriv - 1 - i)

            k = tf.reshape(k, s)
            pp(i,"shape ieme tenseur:",s)
            ks.append(k)

        self.freq_factor=formula(*ks)
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

        self.permut = None
        if self.dim_deriv == 1:
            self.permut = np.arange(len(input_shape))
            self.permut[-1] = axes[0]
            self.permut[axes[0]] = len(input_shape) - 1
        elif self.dim_deriv == 2:
            self.permut = np.arange(len(input_shape))
            self.permut[-2] = axes[0]
            self.permut[-1] = axes[1]
            self.permut[axes[0]] = len(input_shape) - 2
            self.permut[axes[1]] = len(input_shape) - 1
        elif self.dim_deriv == 3:
            self.permut = np.arange(len(input_shape))
            self.permut[-3] = axes[0]
            self.permut[-2] = axes[1]
            self.permut[-1] = axes[2]

            self.permut[axes[0]] = len(input_shape) - 3
            self.permut[axes[1]] = len(input_shape) - 2
            self.permut[axes[2]] = len(input_shape) - 1

        if tuple(self.permut)==tuple(range(len(input_shape))):
            print("pas besoin de permuter")
            self.permut=None


    def fft(self, U):
        if self.dim_deriv == 1:
            return tf.signal.fft(U)
        elif self.dim_deriv == 2:
            return tf.signal.fft2d(U)
        elif self.dim_deriv == 3:
            return tf.signal.fft3d(U)
        else:
            raise Exception("dim must be <=4")

    def ifft(self, U):
        if self.dim_deriv == 1:
            return tf.signal.ifft(U)
        elif self.dim_deriv == 2:
            return tf.signal.ifft2d(U)
        elif self.dim_deriv == 3:
            return tf.signal.ifft3d(U)
        else:
            raise Exception("dim must be <=4")

    def D(self, U):
        if isinstance(U, np.ndarray):
            U = tf.constant(U)

        if self.permut is not None:
            U = tf.transpose(U, self.permut)

        U = pad_nd(U, "smooth_periodizing_padding", self.pads, axes=self.axes_after_transpose)
        U = tf.cast(U,self.dtype_complex)
        U_ = self.fft(U)


        res=[]
        for freq_factor in self.freq_factor:
            U_der=U_*freq_factor
            U_der=self.ifft(U_der)

            if self.suppress_padding_on_output:
                if len(self.pads) == 1:
                    U_der = U_der[..., self.pads[0]:-self.pads[0]]
                elif len(self.pads) == 2:
                    U_der = U_der[..., self.pads[0]:-self.pads[0], self.pads[1]:-self.pads[1]]
                elif len(self.pads) == 3:
                    U_der = U_der[..., self.pads[0]:-self.pads[0], self.pads[1]:-self.pads[1], self.pads[2]:-self.pads[2]]

            if self.permut is not None:
                U_der = tf.transpose(U_der, self.permut)

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

            derivator=_T3FTDerivator_123d([L],F_x.shape,axes=[2],formula=lambda x:x,pad_prop=pad_prop)
            F_x_pred=derivator.D(F)
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


    for pad_prop in [0.1,0.2,0.5,1]:
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

    derivator = _T3FTDerivator_123d(F, [0, 1], Ls,
        formula=lambda a0, a1: {"a0": a0, "a1": a1, "a0^2": a0 ** 2, "a1^2": a1 ** 2,"a0a1":a0*a1})
    DF_fft=derivator.D(F)

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
    #test_precision()
    #test_2d()
    #test_T3FT()
    #test_fft_perf()
    test_create_permut()

import numpy as np

def random_projection(x,k,*arg):
    def gaussian_random_matrix(m,n):
        samp=np.random.normal(0,1,m*n)
        mat=np.reshape(samp,(m,n))
        return mat
    def sparse_random_matrix(m,n):
        x_dist=[-np.sqrt(3),0,np.sqrt(3)]
        y_dist=[1/6.0,2/3.0,1/6.0]
        samp=choices(x_dist,y_dist,k=m*n)
        mat=np.reshape(samp,(m,n))
        return mat
    rw,cl=np.shape(x)
    if arg[0]=='gauss':
        p=gaussian_random_matrix(cl,k)
    elif arg[0]=='sparse':
        p=sparse_random_matrix(cl,k)
    else:
        print('unidentified method')
    y=np.dot(x,p)
    return y
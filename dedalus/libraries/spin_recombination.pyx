
cimport cython

cdef double invsqr2 = 2**(-1/2)

@cython.boundscheck(False)
@cython.wraparound(False)
def recombine_forward_dim3(double [:,:,:,:,::1] input, double [:,:,:,:,::1] output):

    cdef int size0 = input.shape[0]
    cdef int size2 = input.shape[2]
    cdef int size3 = input.shape[3]//2
    cdef int size4 = input.shape[4]
    cdef int i, j, k, l

    for i in range(size0):
        for j in range(size2):
            for k in range(size3):
                for l in range(size4):
                    output[i,0,j,2*k  ,l] = (input[i,1,j,2*k  ,l] + input[i,0,j,2*k+1,l])*invsqr2
                    output[i,1,j,2*k+1,l] = (input[i,1,j,2*k+1,l] + input[i,0,j,2*k  ,l])*invsqr2
                    output[i,1,j,2*k  ,l] = (input[i,1,j,2*k  ,l] - input[i,0,j,2*k+1,l])*invsqr2
                    output[i,0,j,2*k+1,l] = (input[i,1,j,2*k+1,l] - input[i,0,j,2*k  ,l])*invsqr2
                    output[i,2,j,2*k  ,l] =  input[i,2,j,2*k  ,l]
                    output[i,2,j,2*k+1,l] =  input[i,2,j,2*k+1,l]

@cython.boundscheck(False)
@cython.wraparound(False)
def recombine_forward_dim2(double [:,:,:,:,::1] input, double [:,:,:,:,::1] output):

    cdef int size0 = input.shape[0]
    cdef int size2 = input.shape[2]
    cdef int size3 = input.shape[3]//2
    cdef int size4 = input.shape[4]
    cdef int i, j, k, l

    for i in range(size0):
        for j in range(size2):
            for k in range(size3):
                for l in range(size4):
                    output[i,0,j,2*k  ,l] = (input[i,1,j,2*k  ,l] + input[i,0,j,2*k+1,l])*invsqr2
                    output[i,1,j,2*k+1,l] = (input[i,1,j,2*k+1,l] + input[i,0,j,2*k  ,l])*invsqr2
                    output[i,1,j,2*k  ,l] = (input[i,1,j,2*k  ,l] - input[i,0,j,2*k+1,l])*invsqr2
                    output[i,0,j,2*k+1,l] = (input[i,1,j,2*k+1,l] - input[i,0,j,2*k  ,l])*invsqr2


@cython.boundscheck(False)
@cython.wraparound(False)
def recombine_backward_dim3(double [:,:,:,:,::1] input, double [:,:,:,:,::1] output):

    cdef int size0 = input.shape[0]
    cdef int size2 = input.shape[2]
    cdef int size3 = input.shape[3]//2
    cdef int size4 = input.shape[4]
    cdef int i, j, k, l

    for i in range(size0):
        for j in range(size2):
            for k in range(size3):
                for l in range(size4):
                    output[i,0,j,2*k  ,l] = (input[i,1,j,2*k+1,l] - input[i,0,j,2*k+1,l])*invsqr2
                    output[i,0,j,2*k+1,l] = (input[i,0,j,2*k  ,l] - input[i,1,j,2*k  ,l])*invsqr2
                    output[i,1,j,2*k  ,l] = (input[i,0,j,2*k  ,l] + input[i,1,j,2*k  ,l])*invsqr2
                    output[i,1,j,2*k+1,l] = (input[i,0,j,2*k+1,l] + input[i,1,j,2*k+1,l])*invsqr2
                    output[i,2,j,2*k  ,l] =  input[i,2,j,2*k  ,l]
                    output[i,2,j,2*k+1,l] =  input[i,2,j,2*k+1,l]


@cython.boundscheck(False)
@cython.wraparound(False)
def recombine_backward_dim2(double [:,:,:,:,::1] input, double [:,:,:,:,::1] output):

    cdef int size0 = input.shape[0]
    cdef int size2 = input.shape[2]
    cdef int size3 = input.shape[3]//2
    cdef int size4 = input.shape[4]
    cdef int i, j, k, l

    for i in range(size0):
        for j in range(size2):
            for k in range(size3):
                for l in range(size4):
                    output[i,0,j,2*k  ,l] = (input[i,1,j,2*k+1,l] - input[i,0,j,2*k+1,l])*invsqr2
                    output[i,0,j,2*k+1,l] = (input[i,0,j,2*k  ,l] - input[i,1,j,2*k  ,l])*invsqr2
                    output[i,1,j,2*k  ,l] = (input[i,0,j,2*k  ,l] + input[i,1,j,2*k  ,l])*invsqr2
                    output[i,1,j,2*k+1,l] = (input[i,0,j,2*k+1,l] + input[i,1,j,2*k+1,l])*invsqr2


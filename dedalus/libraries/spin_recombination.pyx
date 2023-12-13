
cimport cython

cdef double invsqr2 = 2**(-1/2)


@cython.boundscheck(False)
@cython.wraparound(False)
def recombine_forward(int s, double [:,:,:,:,::1] input, double [:,:,:,:,::1] output):

    cdef int size0 = input.shape[0]
    cdef int size1 = input.shape[1]
    cdef int size2 = input.shape[2]
    cdef int size3 = input.shape[3]
    cdef int size4 = input.shape[4]
    cdef int size3_2 = size3 // 2
    cdef int i, j, k, l, m

    for i in range(size0):
        for j in range(s):
            output[i,j,:,:,:] = input[i,j,:,:,:]
        for k in range(size2):
            for l in range(size3_2):
                for m in range(size4):
                    output[i,s+0,k,2*l  ,m] = (input[i,s+1,k,2*l  ,m] + input[i,s+0,k,2*l+1,m])*invsqr2
                    output[i,s+1,k,2*l+1,m] = (input[i,s+1,k,2*l+1,m] + input[i,s+0,k,2*l  ,m])*invsqr2
                    output[i,s+1,k,2*l  ,m] = (input[i,s+1,k,2*l  ,m] - input[i,s+0,k,2*l+1,m])*invsqr2
                    output[i,s+0,k,2*l+1,m] = (input[i,s+1,k,2*l+1,m] - input[i,s+0,k,2*l  ,m])*invsqr2
        for j in range(s+2, size1):
            output[i,j,:,:,:] = input[i,j,:,:,:]


@cython.boundscheck(False)
@cython.wraparound(False)
def recombine_backward(int s, double [:,:,:,:,::1] input, double [:,:,:,:,::1] output):

    cdef int size0 = input.shape[0]
    cdef int size1 = input.shape[1]
    cdef int size2 = input.shape[2]
    cdef int size3 = input.shape[3]
    cdef int size4 = input.shape[4]
    cdef int size3_2 = size3 // 2
    cdef int i, j, k, l, m

    for i in range(size0):
        for j in range(s):
            output[i,j,:,:,:] = input[i,j,:,:,:]
        for k in range(size2):
            for l in range(size3_2):
                for m in range(size4):
                    output[i,s+0,k,2*l  ,m] = (input[i,s+1,k,2*l+1,m] - input[i,s+0,k,2*l+1,m])*invsqr2
                    output[i,s+0,k,2*l+1,m] = (input[i,s+0,k,2*l  ,m] - input[i,s+1,k,2*l  ,m])*invsqr2
                    output[i,s+1,k,2*l  ,m] = (input[i,s+0,k,2*l  ,m] + input[i,s+1,k,2*l  ,m])*invsqr2
                    output[i,s+1,k,2*l+1,m] = (input[i,s+0,k,2*l+1,m] + input[i,s+1,k,2*l+1,m])*invsqr2
        for j in range(s+2, size1):
            output[i,j,:,:,:] = input[i,j,:,:,:]


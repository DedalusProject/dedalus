
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def cross_product(double [:,::1] data0, double [:,::1] data1, double [:,::1] output):

    cdef int size1 = data0.shape[1]
    cdef int i
    
    for i in range(size1):
        output[0,i] = data0[2,i]*data1[1,i] - data0[1,i]*data1[2,i]
        output[1,i] = data0[0,i]*data1[2,i] - data0[2,i]*data1[0,i]
        output[2,i] = data0[1,i]*data1[0,i] - data0[0,i]*data1[1,i]

@cython.boundscheck(False)
@cython.wraparound(False)
def num_field_product(double number, double [::1] input, double [::1] output):

    cdef int size = input.shape[0]
    cdef int i

    for i in range(size):
        output[i] = number*input[i]

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_product_inplace(double number, double [::1] input, double [::1] output):

    cdef int size = input.shape[0]
    cdef int i

    for i in range(size):
        output[i] = output[i] + number*input[i]

@cython.boundscheck(False)
@cython.wraparound(False)
def sum(double [::1] data0, double [::1] data1, double [::1] output):

    cdef int size = data0.shape[0]
    cdef int i

    for i in range(size):
        output[i] = data0[i] + data1[i]

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_inplace(double [::1] input, double [::1] output):

    cdef int size = input.shape[0]
    cdef int i

    for i in range(size):
        output[i] += input[i]



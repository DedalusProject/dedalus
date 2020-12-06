
# tuple helper functions
dual    =             lambda t: tuple(-e for e in t)
apply   = lambda p:   lambda t: tuple(t[int(i)] for i in p)
sum_    = lambda k:   lambda t: sum(t[int(i)] for i in k if 0 <= i < len(t))
remove  = lambda k:   lambda t: tuple(s for i,s in enumerate(t) if not i in k)
replace = lambda j,n: lambda t: tuple(s if i!=j else n for i,s in enumerate(t))

# converts integer args to tuple singleton i --> (i,).
def int2tuple(func):
    if func.__name__ == '__getitem__':
        def wrapper(*args):
            self, args = args[0], args[1]
            if type(args) == slice: return func(self,(args,))
            if type(args) ==   int: return func(self,(args,))
            args = (self,tuple((s,) if type(s)==int else s for s in args))
            return func(*args)
        return wrapper
    return lambda *args: func(*tuple((s,) if type(s)==int else s for s in args))
        
# forward and inverse mapping between spin/regularity elements and integers.
def tuple2index(tup,indexing):
    return int('0'+''.join(str(indexing.index(s)) for s in tup),len(indexing))

def index2tuple(index,rank,indexing):
    s = np.base_repr(index,len(indexing),rank)
    return apply(s[(rank==0)-rank:])(indexing)

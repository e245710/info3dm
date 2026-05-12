import numpy as np

def true_function(x):
    """
    真の関数 y = sin(pi * x * 0.8) * 10
    
    >>> true_function(np.array([0.0]))
    array([0.])
    """
    return np.sin(np.pi * x * 0.8) * 10

if __name__ == "__main__":
    import doctest
    # doctestを実行し、x=0 のときに y=0 となることを確認
    doctest.testmod()
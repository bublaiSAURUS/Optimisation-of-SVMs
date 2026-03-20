import numpy as np

def create_demo():

    # 2 points for contour plot
    X = np.array([
    [1.,0.],
    [-1.,0.]
    ])

    y = np.array([1.,-1.])
    
    # 4 points for Q-convergence:
    # X = np.array([
    #     [1.,0.],
    #     [0.,1.],
    #     [-1.,0.],
    #     [0.,-1.]
    # ])

    # y = np.array([1.,1.,-1.,-1.])

    return X, y



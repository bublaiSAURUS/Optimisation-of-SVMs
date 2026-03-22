import numpy as np

class SMO_MOD:
    def __init__(self, X, y, C=1, kernel = None, max_iter = 200, tol = 1e-6, eps = 1e-8):
        self._point = X
        self._target = y
        self.m, self.n = np.shape(self._point)
        self._C = C
        self.fcache = np.zeros(self.m)
        self.alphas = np.zeros(self.m)
        self.kernel = kernel
        self.tol = tol
        self.eps = eps
        self.w = np.zeros(self.n)
        self.max_iter = max_iter
        self.iter = 0

    def predict(self, x, b):
        result = np.matmul((self.alphas * self._target) , self.kernel(self._point, x)) + b
        return result


    def compute_F(self, i2):
        res = np.zeros(self.m)
        for i in range(self.m):
            res[i] = self.kernel(self._point[i2, :], self._point[i, :])
        u = np.dot(self.alphas * self._target, res)
        return u - self._target[i2]


    def takeStep(self, i1, i2):
        if (i1 == i2):
            return 0
        alph1 = self.alphas[i1]
        alph2 = self.alphas[i2]
        y1 = self._target[i1]
        y2 = self._target[i2]
        F1 = self.fcache[i1]
        F2 = self.fcache[i2]
        s = y1*y2

        if y1 != y2:
            L = max(0, alph2 - alph1)
            H = min(self._C, self._C + alph2 - alph1)
        else:
            L = max(0, alph2 + alph1 - self._C)
            H = min(self._C, alph2 + alph1)
        
        if L == H:
            return 0
        
        x1 = self._point[i1, :]            
        x2 = self._point[i2, :]
        
        k11 = self.kernel(x1, x1)
        k12 = self.kernel(x1, x2)
        k22 = self.kernel(x2, x2)
        eta = 2*k12 - k11 - k22
        if eta < 0:
            a2 = alph2 - y2*(F1 - F2)/eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            a1_L = alph1 + s*(alph2 - L)
            a1_H = alph1 + s*(alph2 - H)
            Lobj = -(a1_L + L) + 0.5 * (k11 * a1_L**2 + k22 * L**2 + 2 * s * k12 * a1_L * L)
            Hobj = -(a1_H + H) + 0.5 * (k11 * a1_H**2 + k22 * H**2 + 2 * s * k12 * a1_H * H)
            if (Lobj > Hobj + self.eps):
                a2 = L
            elif (Lobj < Hobj  - self.eps):
                a2 = H
            else:
                a2 = alph2
        
        if (abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps)):
            return 0
        
        a1 = alph1 + s * (alph2 - a2)
        
        if self.kernel.__name__ == "linear_kernel":
            self.w += y1 * (a1 - alph1) * x1 + y2 * (a2 - alph2) * x2


        #TODO: Update fcache[i] for i in I_0 using new Lagrange Multipliers
        # Manually doing this:
        del_alph1 = a1 - alph1
        del_alph2 = a2 - alph2
        for i in range(self.m):
            if ((0 < self.alphas[i] and self.alphas[i] < self._C)):
                self.fcache[i] += self._target[i1]*del_alph1*self.kernel(self._point[i1, :], self._point[i, :]) + self._target[i2]*del_alph2*self.kernel(self._point[i2, :], self._point[i, :])

        self.alphas[i1] = a1
        self.alphas[i2] = a2
        
        #TODO: Update I_0, I_1, I_2, I_3, I_4 (Not needed as I am manually checking)

        self.fcache[i1] = F1 + y1 * (a1 - alph1)*k11 + y2 * (a2 - alph2)*k12
        self.fcache[i2] = F2 + y1 * (a1 - alph1)*k12 + y2 * (a2 - alph2)*k22

        #TODO: Compute (i_low, b_low) and (i_up, b_up) using eqns (11a) and (11b) and 3. from sec. 5
        for i in range(self.m):
            if ((0 < self.alphas[i] and self.alphas[i] < self._C) or i == i1 or i == i2):
                if self.fcache[i] > self.b_low:
                    self.b_low = self.fcache[i]
                    self.i_low = i
                if self.fcache[i] < self.b_up:
                    self.b_up = self.fcache[i]
                    self.i_up = i
        return 1    


# I_0 = {i : 0 < alpha_i<C}
# I_1 = {i : y_i = 1, alpha_i = 0}
# I_2 = {i : y_i = -1, alpha_i = C}
# I_3 = {i : y_i = 1, alpha_i = C}
# I_4 = {i : y_i = -1, alpha_i = 0}

    def examineExample(self, i2):
        y2 = self._target[i2]
        alph2 = self.alphas[i2]
        # if (i2 in I_0):
        if ((0 < self.alphas[i2] and self.alphas[i2] < self._C)):
            F2 = self.fcache[i2]
        else:
            
            #TODO: Compute F2 = F_i2
            F2 = self.compute_F(i2)

            self.fcache[i2] = F2
            # if (((i2 in I_1) or (i2 in I2)) and (F2 < self.b_up)):
            if ((self._target[i2] == 1 and self.alphas[i2] == 0) or (self._target[i2] == -1 and self.alphas[i2] == self._C) and (F2 < self.b_up)):
                self.b_up = F2
                self.i_up = i2
            
            # elif (((i2 in I_3) or (i2 in I_4)) and (F2 > self.b_low)):
            elif ((self._target[i2] == 1 and self.alphas[i2] == self._C) or (self._target[i2] == -1 and self.alphas[i2] == 0) and (F2 > self.b_low)):    
                self.b_low = F2
                self.i_low = i2
        
        optimality = 1
        # if ((i2 in I_0) or (i2 in I_1) or (i2 in I_2)):
        if((0 < self.alphas[i2] and self.alphas[i2] < self._C) or (self._target[i2] == 1 and self.alphas[i2] == 0) or (self._target[i2] == -1 and self.alphas[i2] == self._C)):
            if (self.b_low - F2 > 2 * self.tol):
                optimality = 0
                i1 = self.i_low
        # if ((i2 in I_0) or (i2 in I_3) or (i2 in I_4)):
        if((0 < self.alphas[i2] and self.alphas[i2] < self._C) or (self._target[i2] == 1 and self.alphas[i2] == self._C) or (self._target[i2] == -1 and self.alphas[i2] == 0)):
            if (F2 - self.b_up > 2 * self.tol):
                optimality = 0
                i1 = self.i_up
        if optimality == 1:
            return 0
        
        # if (i2 in I_0):
        if ((0 < self.alphas[i2] and self.alphas[i2] < self._C)):
            if (self.b_low - F2 > F2 - self.b_up):
                i1 = self.i_low
            else:
                i1 = self.i_up
        
        if self.takeStep(i1, i2):
            return 1
        else:
            return 0
    
    
    
    def fit(self):
        self.b_up = -1
        
        #TODO: initialise i_up = any one index of class 1
        self.i_up = np.where(self._target == 1)[0][0]
                
        self.b_low = 1
        
        #TODO: initialise i_low = any one index of class 2  
        self.i_low = np.where(self._target == -1)[0][0]
        
        self.fcache[self.i_low] = 1
        self.fcache[self.i_up] = -1

        counter = 0
        numChanged = 0
        examineAll = 1
        while ((numChanged > 0 or examineAll) and self.iter < self.max_iter):
            numChanged = 0
            self.iter += 1
            print(self.iter)
            if examineAll:
                for i in range(self.m):
                    numChanged += self.examineExample(i)
            else:
                inner_loop_success = 1
                while True:
                    if ((self.b_up > self.b_low - 2*self.tol) or inner_loop_success == 0):
                        break
                    i2 = self.i_low
                    y2 = self._target[i2]
                    alph2 = self.alphas[i2]
                    i1 = self.i_up
                    inner_loop_success = self.takeStep(self.i_up, self.i_low)
                    numChanged += inner_loop_success
                
                numChanged = 0
            
            if examineAll == 1:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1
        return self.alphas

import numpy as np

class SMO:
    def __init__(self, X, y, C=1, kernel='linear', b=0, max_iter=300, tol=1e-5, eps=1e-8):
        self._point = X
        self._target = y
        self.m, self.n = np.shape(self._point)
        self.C = C

        self.alphas = np.zeros(self.m)
        self.b = b

        self.kernel = kernel 
        self.error_cache = np.zeros(self.m)

        self.max_iter=max_iter
        self.tol = tol
        self.eps = eps
        self.alpha_history = [self.alphas.copy()]
        
        # for linear kernel
        self.w = np.zeros(self.n) 

        # for experiments
        self.obj_history = []

    def predict(self, x):
        result = np.matmul((self.alphas * self._target) , self.kernel(self._point, x)) + self.b
        return result

    def get_error(self, i):
        return self.predict(self._point[i,:]) - self._target[i]


    def compute_objective(self):
        # obj = 0.0
        obj_term = 0.0
        ay = self.alphas * self._target
        # for i in range(self.m):
        #     for j in range(self.m):
        #         obj += 0.5 * self.alphas[i] * self.alphas[j] * \
        #             self._target[i] * self._target[j] * \
        #             self.kernel(self._point[i], self._point[j])
        # obj -= np.sum(self.alphas)    
        # return obj  
        for i in range(self.m):
            ki = self.kernel(self._point[i], self._point)
            obj_term += ay[i] * np.dot(ki, ay)
        return 0.5 * obj_term - np.sum(self.alphas)

    def takeStep(self, i1, i2):
        if (i1 == i2):
            return 0
        
        alph1 = self.alphas[i1]
        alph2 = self.alphas[i2]
        y1 = self._target[i1]
        E1 = self.error_cache[i1]
        y2 = self._target[i2]
        s = y1 * y2

        if y1 != y2:
            L = max(0, alph2 - alph1)
            H = min(self.C, self.C + alph2 - alph1)
        else:
            L = max(0, alph2 + alph1 - self.C)
            H = min(self.C, alph2 + alph1)

        if L == H:
            return 0

        x1 = self._point[i1, :]
        x2 = self._point[i2, :]
        
        k11 = self.kernel(x1, x1)
        k12 = self.kernel(x1, x2)
        k22 = self.kernel(x2, x2)

        eta = k11 + k22 - 2 * k12

        E2 = self.error_cache[i2]

        if eta > 0:
            a2 = alph2 + y2 * (E1 - E2) / eta
            if (a2 <= L):
                a2 = L
            elif (a2 >= H):
                a2 = H
        else:
            a1_L = alph1 + s*(alph2 - L)
            a1_H = alph1 + s*(alph2 - H)
            Lobj = (a1_L + L) - 0.5 * (k11 * a1_L**2 + k22 * L**2 + 2 * s * k12 * a1_L * L)
            Hobj = (a1_H + H) - 0.5 * (k11 * a1_H**2 + k22 * H**2 + 2 * s * k12 * a1_H * H)

            if (Lobj < Hobj - self.eps):
                a2 = L
            elif (Lobj > Hobj  + self.eps):
                a2 = H
            else:
                a2 = alph2
        
        if (abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps)):
            return 0
        
        a1 = alph1 + s * (alph2 - a2)

        b_old = self.b
        
        # Update threshold
        b1 = self.b - (E1 + y1*(a1 - alph1)*k11 + y2*(a2 - alph2)*k12)
        b2 = self.b - (E2 + y1*(a1 - alph1)*k12 + y2*(a2 - alph2)*k22) 
        if 0 < a1 < self.C:
            self.b = b1
        elif 0 < a2 < self.C:
            self.b = b2
        else:
            self.b = np.mean([b1, b2])


        if self.kernel.__name__ == "linear_kernel":
            self.w += y1 * (a1 - alph1) * x1 + y2 * (a2 - alph2) * x2 


        #implement error cache

        for i in range(self.m):
            self.error_cache[i] += (
                y1 * (a1 - alph1) * self.kernel(self._point[i1], self._point[i]) +
                y2 * (a2 - alph2) * self.kernel(self._point[i2], self._point[i]) +
                (self.b - b_old)
            )


        self.alphas[i1] = a1
        self.alphas[i2] = a2
        self.alpha_history.append(self.alphas.copy())

        return 1
    

    def examineExample(self, i2):
        y2 = self._target[i2]
        alpha2 = self.alphas[i2]
        E2 = self.error_cache[i2]
        r2 = E2 * y2

        if ((r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0)):
            if len(self.alphas[(self.alphas != 0) & (self.alphas != self.C)]) > 1:
                if E2 > 0:
                    i1 = np.argmin(self.error_cache)
                else:
                    i1 = np.argmax(self.error_cache)
                if self.takeStep(i1, i2):
                    return 1
            
            i1_list = [i for i, alpha in enumerate(self.alphas) if alpha != 0 and alpha != self.C]
            i1_list = np.roll(i1_list, np.random.choice(np.arange(self.m)))
            for i1 in i1_list:
                if self.takeStep(i1, i2):
                    return 1

            i1_list = np.roll(np.arange(self.m), np.random.choice(np.arange(self.m)))
            for i1 in i1_list:
                if self.takeStep(i1, i2):
                    return 1

        return 0

    def fit(self):
        for i in range(self.m):
            self.error_cache[i] = self.get_error(i)
        numChanged = 0
        examineAll = 1
        while numChanged > 0 or examineAll:
            numChanged = 0
            if examineAll:
                for i in range(self.m):
                    numChanged += self.examineExample(i)
            else:
                i_list = [i for i, alpha in enumerate(self.alphas) if 0 < alpha and alpha < self.C]
                for i in i_list:
                    numChanged += self.examineExample(i)

            if examineAll == 1:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1
        return self.alphas, self.b
        
from numpy import cos, sin
import numpy as np
class DPendulum:
    #http://scienceworld.wolfram.com/physics/DoublePendulum.html
    def __init__(self, l1=1.0, l2=1.0, m1=1.0, m2=1.0, theta1 = -1., theta2 = 0.5):
        #Set params
        self.coupled = []
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.g = 9.81
        #Set initial conditions
        self.theta1 = theta1 + 0.05*np.random.randn()
        self.theta2 = theta2 + 0.05*np.random.randn()
        self.p1     = 0
        self.p2     = 0
    def __C1(self):
        tmp  = self.p1 * self.p2 * sin(self.theta1 - self.theta2)
        tmp /= self.l1 * self.l2  * (self.m1 + self.m2 * sin(self.theta1 - self.theta2)**2)
        return tmp
    def __C2(self):
        tmp  = self.l2**2 * self.m2 * self.p1**2 + self.l1**2 * (self.m1 + self.m2) * self.p2**2 
        tmp -= self.l1 * self.l2 * self.m2 * self.p1 * self.p2 * cos(self.theta1 - self.theta2)
        tmp /= 2* (self.l1 * self.l2 * (self.m1 + self.m2 * sin(self.theta1 - self.theta2)**2))**2
        tmp *= sin(2*(self.theta1 - self.theta2))
        return tmp
    def dtheta1(self):
        tmp  = self.l2 * self.p1 - self.l1 * self.p2 * cos(self.theta1 - self.theta2) 
        tmp /= self.l1**2 * self.l2 * (self.m1 + self.m2 * sin(self.theta1 - self.theta2)**2)
        return tmp
    def dtheta2(self):
        tmp  = self.l1*(self.m1 + self.m2)* self.p2-self.l2 * self.m2 * self.p1 * cos (self.theta1 - self.theta2)
        tmp /= self.l1 * self.l2**2 * self.m2 * (self.m1 + self.m2 * sin(self.theta1 - self.theta2)**2)
        return tmp
    def dp1(self):
        coupl_term = 0
        if len(self.coupled) > 0:
            for (w, dpend) in self.coupled:
                coupl_term += w * 2 * (self.theta1 - dpend.theta1)
        return -(self.m1 + self.m2) * self.g * self.l1 * sin(self.theta1) - self.__C1() + self.__C2() - coupl_term
    def dp2(self):
        return - self.m2 * self.g * self.l2 * sin(self.theta2) + self.__C1() - self.__C2()
    
    def leapfrog_step(self,tau):
        dp1 = self.dp1()
        dp2 = self.dp2()
        self.p1 += dp1 * tau / 2
        self.p2 += dp2 * tau / 2
        
        dtheta1 = self.dtheta1()
        dtheta2 = self.dtheta2()
        self.theta1 += dtheta1 * tau
        self.theta2 += dtheta2 * tau
        
        dp1 = self.dp1()
        dp2 = self.dp2()
        self.p1 += dp1 * tau / 2
        self.p2 += dp2 * tau / 2

    def couple(self, dpend, w):
        self.coupled += [(w, dpend)]


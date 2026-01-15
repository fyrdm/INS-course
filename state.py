import numpy as np
import matplotlib.pylab as plt

from ahrs import Quaternion
#
# System state is position, velocity and acceleration in 3D
#


class SystemState(object):
    def __init__(self, x_0, P_0)->None:
        self._x = x_0 # initial state
        self._P = P_0
        
    @property
    def x(self):
        return self._x


    @property
    def P(self):
        return self._P

#
# Class to represent system with its dynamics
#
class System(object):
    def __init__(self, initial_state, jerk_variance):
        self.state_plus = initial_state # updated/initial
        self.state_minus = None # propagated state
        self.jerk_variance = jerk_variance
    
    def _get_state_transition(self, dt):
        return np.array([
                [1,0,0,  dt, 0, 0,   1/2*dt**2, 0,0],
                [0,1,0,  0, dt, 0,   0, 1/2*dt**2,0],
                [0,0,1,  0,0, dt,    0, 0, 1/2*dt**2],

                [0,0,0, 1,0,0 ,dt, 0, 0],
                [0,0,0, 0,1,0 ,0,dt,  0],
                [0,0,0, 0,0,1 ,0,0,  dt],

                
                [0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,1],
            ])
    
    
    def propagate(self, dt):
        # x_t+1_minus = phi * x_t
        Phi_k = self._get_state_transition(dt)
        
        # Propagate State
        x_minus =  np.dot(Phi_k , self.state_plus.x)
        # Propagate Covariance
        P_minus  = Phi_k @ self.state_plus.P @ Phi_k.T + self.Q(dt)
        
        self.state_minus = SystemState(x_minus, P_minus)
    
    # System noise 
    def Q(self, dt):
        return self.jerk_variance * np.vstack([
                np.hstack([dt**5/20 *np.eye(3), dt**4/8*np.eye(3), dt**3/6*np.eye(3)]),
                np.hstack([dt**4/8*np.eye(3), dt**3/3*np.eye(3), dt**2/2*np.eye(3)]),
                np.hstack([dt**3/6*np.eye(3), dt**2/2*np.eye(3), dt*np.eye(3)])
            ])
        
    
    def update_wo_measurement(self):
        self.state_plus = self.state_minus
        self.state_minus = None

    def update(self, z, H, R):
        
        
        # z = [a_x, a_y, a_z]
        
        
        if (z.shape[0] ==2):
            print("Here")
        dz = z - np.dot(H, self.state_minus.x) # 3.54>
        
        M = np.dot(H, np.dot(self.state_minus.P, H.T)) + R 
        
        M_inv = np.linalg.inv(M)
        
        K = np.dot(self.state_minus.P, np.dot(H.T, M_inv)) # Eq. 3.60 > 
        
        x_plus = self.state_minus.x + np.dot(K, dz)
        
        P_plus = self.state_minus.P - np.dot(K, np.dot(H, self.state_minus.P))
        
        self.state_plus = SystemState(x_plus, P_plus)
        self.state_minus = None
        
        
        
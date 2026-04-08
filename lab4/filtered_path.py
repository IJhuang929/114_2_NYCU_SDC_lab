import numpy as np
import matplotlib.pyplot as plt
from math import atan
from kalman_filter import KalmanFilter

class virtual_path:
    def __init__(self, frame):
        self.frame = frame
        self.ideal_path = {'x':[], 'y':[], 'yaw':[]}
        self.real_path = {'x':[], 'y':[], 'yaw':[]}
        self.control = {'x':[], 'y':[], 'yaw':[]}
        self.measurement = {'x':[], 'y':[]}
        
        self.initial_x = 0
        self.initial_y = 0
        self.initial_vx = 2
        self.initial_vy = 20
        self.initial_yaw = atan(self.initial_vy/self.initial_vx) 
        self.ay = -1
        
        self.control_var = 1
        self.measurement_var = 3

        self.create_ideal(self.frame)
        
    def create_ideal(self, frame):
        x = self.initial_x
        y = self.initial_y
        yaw = self.initial_yaw
        vx = self.initial_vx
        vy = self.initial_vy
        ay = self.ay
        
        self.ideal_path = {'x':[x], 'y':[y], 'yaw':[yaw]}
        
        for i in range(frame):
            x += vx
            y += vy
            self.ideal_path['x'].append(x)
            self.ideal_path['y'].append(y)
            vy += ay
            self.ideal_path['yaw'].append(atan(vy/vx))
    
    def create_real(self, frame):
        x = self.initial_x
        y = self.initial_y
        yaw = self.initial_yaw
        last_yaw = yaw
        vx = self.initial_vx
        vy = self.initial_vy
        vyaw = None
        ay = self.ay
    
        self.real_path = {'x':[x], 'y':[y], 'yaw':[yaw]}
        self.control = {'x':[], 'y':[], 'yaw':[]}
        self.measurement = {'x':[], 'y':[]}
        
        for i in range(frame):
            
            self.control['x'].append(vx)
            self.control['y'].append(vy)

            x_noise = np.random.normal(0, self.control_var, 1)[0]
            y_noise = np.random.normal(0, self.control_var, 1)[0]
            x += vx + x_noise
            y += vy + y_noise
            
            self.real_path['x'].append(x)
            self.real_path['y'].append(y)
            
            measure_x = x + np.random.normal(0, self.measurement_var, 1)[0]
            measure_y = y + np.random.normal(0, self.measurement_var, 1)[0]
            
            self.measurement['x'].append(measure_x)
            self.measurement['y'].append(measure_y)
            
            vy += ay

            yaw = atan((vy + y_noise)/(vx + x_noise))
            vyaw = yaw - last_yaw
            last_yaw = yaw
            self.control['yaw'].append(vyaw)
            
    def visualize(self, path):
        l = range(self.frame)
        plt.plot(path['x'], path['y'])
        plt.show()

def main():
    vp = virtual_path(41)
    vp.create_real(vp.frame)
    kf = KalmanFilter(yaw=atan(vp.initial_vy/vp.initial_vx))
    filtered_path = {'x':[kf.state[0]], 'y':[kf.state[1]], 'yaw':[kf.state[2]]}
    for i in range(vp.frame):
        u = np.array([vp.control['x'][i], vp.control['y'][i], vp.control['yaw'][i]])
        kf.predict(u)
        z = np.array([vp.measurement['x'][i], vp.measurement['y'][i]])
        kf.update(z)
        
        filtered_path['x'].append(kf.state[0])
        filtered_path['y'].append(kf.state[1])
        filtered_path['yaw'].append(kf.state[2])
    
    plt.plot(vp.ideal_path['x'], vp.ideal_path['y'], alpha=0.5, c="r", label="ideal path")
    plt.plot(vp.real_path['x'], vp.real_path['y'], alpha=0.5, c="g", label="real path")
    plt.plot(filtered_path['x'], filtered_path['y'], alpha=0.5, c="b", label="filtered path")
    plt.scatter(vp.measurement['x'], vp.measurement['y'], s=10, c='b', label="measurement")
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
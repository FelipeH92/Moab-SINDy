import numpy as np

g = 9.81
A = 2
f = 6

theta_ramp_x = 5*np.pi/180
theta_ramp_y = 5*np.pi/180
target = 0.5

x_step = theta_ramp_x/target
y_step = theta_ramp_x/target

def sin6tpi(t, stack = False):
    if stack:
        return np.column_stack((A*np.sin(f*t + np.pi))*np.pi/180)
    return (A*np.sin(f*t + np.pi))*np.pi/180

def sin6tpi2(t, stack = False):
    if stack:
        return np.column_stack((A*np.sin(f*t + (np.pi/2)))*np.pi/180)
    return (A*np.sin(f*t + (np.pi/2)))*np.pi/180

def ramp_x(t, stack = False):
    if stack:
        rampList = []
        for i in t:
            if i < target:
                rampList.append(x_step*i)
            else:
                rampList.append(theta_ramp_x)
        rampList = np.column_stack(np.asarray(rampList))
        return rampList
    if t < target:
        return x_step*t
    return theta_ramp_x

def ramp_y(t, stack = False):
    if stack:
        rampList = []
        for i in t:
            if i < target:
                rampList.append(y_step*i)
            else:
                rampList.append(theta_ramp_y)
        rampList = np.column_stack(np.asarray(rampList))
        return rampList
    if t < target:
        return y_step*t
    return theta_ramp_y


def moab_simplified(f, t, thetaxFunc, thetayFunc):
    #_, xdot, _, ydot = f
    thetax = thetaxFunc(t)
    thetay = thetayFunc(t)
    xdot2 = ((3/5)*g)*thetay
    ydot2 = ((3/5)*g)*thetax
    return [xdot2, ydot2]

mb = 0.0027
rb = 0.02
rs = rb - 0.0002
Jb = ((2/5) * mb * (rb**5 - rs**5) )/ (rb**3 - rs**3)
dt = 0.001
aux = mb + (Jb/(rb**2))

def derivative(ub, ua):
    return (ua - ub)/2*dt

def sin6tpidt(t, stack = False):
    if stack:
        input_stack = np.zeros((t.shape[0],2))
        ub = (A*np.sin(f*t[0] + np.pi))*np.pi/180
        input_stack[0,0] = ub
        for i in range(1,t.shape[0]-1):
            u = (A*np.sin(f*t[i] + np.pi))*np.pi/180
            ua = (A*np.sin(f*t[i+1] + np.pi))*np.pi/180
            du = derivative(ub,ua)
            ub = u
            input_stack[i,0] = u
            input_stack[i,1] = du
        input_stack[-1,0] = (A*np.sin(f*t[-1] + np.pi))*np.pi/180
        ua = (A*np.sin(f*(t[-1]+dt) + np.pi))*np.pi/180
        input_stack[-1,1] = derivative(ub,ua)
        return input_stack
    u = (A*np.sin(f*t + np.pi))*np.pi/180
    if (t == 0):
        du = 0
    else:
        ub = (A*np.sin(f*(t-dt) + np.pi))*np.pi/180
        ua = (A*np.sin(f*(t+dt) + np.pi))*np.pi/180
        du = derivative(ub,ua)
    return [u, du]

def sin6tpi2dt(t, stack = False):
    if stack:
        input_stack = np.zeros((t.shape[0],2))
        ub = (A*np.sin(f*t[0] + (np.pi/2)))*np.pi/180
        input_stack[0,0] = ub
        for i in range(1,t.shape[0]-1):
            u = (A*np.sin(f*t[i] + (np.pi/2)))*np.pi/180
            ua = (A*np.sin(f*t[i+1] + (np.pi/2)))*np.pi/180
            du = derivative(ub,ua)
            ub = u
            input_stack[i,0] = u
            input_stack[i,1] = du
        input_stack[-1,0] = (A*np.sin(f*t[-1] + (np.pi/2)))*np.pi/180
        ua = (A*np.sin(f*(t[-1]+dt) + (np.pi/2)))*np.pi/180
        input_stack[-1,1] = derivative(ub,ua)
        return input_stack
    u = (A*np.sin(f*t + (np.pi/2)))*np.pi/180
    if (t == 0):
        du = 0
    else:
        ub = (A*np.sin(f*(t-dt) + (np.pi/2)))*np.pi/180
        ua = (A*np.sin(f*(t+dt) + (np.pi/2)))*np.pi/180
        du = derivative(ub,ua)
    return [u, du]

'''
    f = [x, xdot, y, ydot]
'''
def moab(f, t, thetaxFunc, thetayFunc):
    thetax, thetaxdot = thetaxFunc(t)
    thetay, thetaydot = thetayFunc(t)
    xdot2 = (mb*g*np.sin(thetay) - mb*(f[0]*(thetaydot**2) + f[2]*thetaxdot*thetaydot))/aux
    ydot2 = (mb*g*np.sin(thetax) - mb*(f[2]*(thetaxdot**2) + f[0]*thetaxdot*thetaydot))/aux

    return [f[1],xdot2,f[3],ydot2]
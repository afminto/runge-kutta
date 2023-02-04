import numpy as np
import matplotlib.pyplot as plt

def vector_norm(vector):
    sum_sq = 0
    for comp in vector:
        sum_sq += (comp ** 2)
    return sum_sq ** (1/2)

def euler_advance(positions, velocities, h, step_no, N):
    for i in range(N):
        positions[i,step_no,:] = positions[i,step_no-1,:] + h * velocities[i,step_no-1,:]
        velocities[i,step_no,:] = velocities[i,step_no-1,:] + h * lennard_jones_accel(positions[:,step_no-1,:], i, N)
        
def rk2_advance(positions, velocities, h, step_no, N):
    half_pos = np.zeros(shape=(N,3))
    half_velocities = np.zeros(shape=(N,3))
    for i in range(N):
        half_pos[i] = positions[i,step_no - 1,:] + (h / 2) * velocities[i,step_no - 1,:]  #x + q1/2
        half_velocities[i] = velocities[i,step_no-1,:] + (h / 2) * lennard_jones_accel(positions[:,step_no-1,:], i, N) #v + p1/2
    for i in range(N):
        positions[i, step_no, :] = positions[i, step_no - 1, :] + h * half_velocities[i,:] #x + q2
        velocities[i, step_no, :] = velocities[i,step_no - 1,:] + h * lennard_jones_accel(half_pos, i, N) #v + p2
        
def rk4_advance(positions, velocities, h, step_no, N):
    q1 = h * velocities[:,step_no - 1,:]
    p1 = np.zeros(shape = (N,3))
    for i in range(N):
        p1[i,:] = h * lennard_jones_accel(positions[:,step_no-1,:], i, N)
    input_pos_2 = positions[:,step_no-1,:] + (1/2)*q1
    input_vel_2 = velocities[:,step_no-1,:] + (1/2)*p1
    q2 = h * input_vel_2
    p2 = np.zeros(shape = (N, 3))
    for i in range(N):
        p2[i,:] = h * lennard_jones_accel(input_pos_2, i, N)
    
    input_pos_3 = positions[:,step_no-1,:] + (1/2)*q2
    input_vel_3 = velocities[:,step_no-1,:] + (1/2)*p2
    q3 = h * input_vel_3
    p3 = np.zeros(shape = (N, 3))
    for i in range(N):
        p3[i,:] = h * lennard_jones_accel(input_pos_3, i, N)
    
    
    input_pos_4 = positions[:,step_no-1,:] + q3
    input_vel_4 = velocities[:,step_no-1,:] + p3
    q4 = h * input_vel_4
    p4 = np.zeros(shape = (N, 3))
    for i in range(N):
        p4[i,:] = h * lennard_jones_accel(input_pos_4, i, N)
        
    x_displacement = (1 / 6) * (q1 + (2 * q2) + (2 * q3) + q4)
    v_displacement = (1 / 6) * (p1 + (2 * p2) + (2 * p3) + p4)
    
    positions[:,step_no,:] = positions[:,step_no - 1,:] + x_displacement
    velocities[:,step_no,:] = velocities[:,step_no - 1,:] + v_displacement
    

def lennard_jones_accel(positions, i, N):
    accel_vector = np.zeros(3)
    for obj in range(N):
        if obj == i:
            continue
        else:
            x = positions[obj,:] - positions[i,:]
            normx = vector_norm(x)
            if normx > 3:
                continue
            factor = (-1 * (normx ** -14)) + (0.5 * (normx ** -8))
            accel_obj = factor * x
            accel_vector += accel_obj
            
    return accel_vector
    
    
def simulate(initial_x, initial_v, h, N, time, method):
    num_steps = round(time / h)
    positions = np.zeros(shape=(N,num_steps,3))
    velocities = np.zeros(shape=(N,num_steps,3))
    positions[:,0,:] = initial_x
    velocities[:,0,:] = initial_v
    for step_no in range(1, num_steps):
        if method == 'euler':
            euler_advance(positions, velocities, h, step_no, N)
        elif method == 'rk2':
            rk2_advance(positions, velocities, h, step_no, N)
        elif method == 'rk4':
            rk4_advance(positions, velocities, h, step_no, N)
    
    return positions, velocities
    
init_pos = [[0,0,0]]
init_vel = [[0.1,0,0]]
methods = ['euler', 'rk2', 'rk4']
for method in methods:
    positions, velocities = simulate(init_pos, init_vel, 1, 1, 10, method)
    x_pos = positions[0,:,0]
    y_pos = positions[0,:,1]
    times = np.arange(10)
    plt.plot(times, x_pos, markevery=1)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(f'{method} method for single Argon Atom')
    plt.show()
    plt.clf()
    plt.plot(x_pos, y_pos)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{method} method for single Argon Atom')
    plt.show()
    plt.clf()
    
def x_momentum_ratio(init_vel, final_vel, N):
    init_x_momentum = 0
    final_x_momentum = 0
    for i in range(N):
        init_x_momentum += init_vel[i][0]
        final_x_momentum += final_vel[i,0]
    ratio = init_x_momentum / final_x_momentum
    return ratio

def kinetic_energy_ratio(init_vel, final_vel, N):
    init_kinetic = 0
    final_kinetic = 0
    for i in range(N):
        init_kinetic += (vector_norm(init_vel[i]) ** 2)
        final_kinetic += (vector_norm(final_vel[i,:]) ** 2)
    ratio = init_kinetic / final_kinetic
    return ratio
    

init_pos = [[-5,0,0], [0,0,0]]
init_vel = [[0.2,0,0],[0,0,0]]
methods = ['euler', 'rk2', 'rk4']
steps = [1, 0.1, 0.01]
for method in methods:
    for step in steps:
        positions, velocities = simulate(init_pos, init_vel, step, 2, 40, method)
        x_pos_0 = positions[0,:,0]
        y_pos_0 = positions[0,:,1]
        x_pos_1 = positions[1,:,0]
        y_pos_1 = positions[1,:,1]
        x_vel_0 = velocities[0,:,0]
        x_vel_1 = velocities[1,:,0]
        times = np.arange(0, 40, step)
        plt.plot(x_pos_0, y_pos_0)
        plt.plot(x_pos_1, y_pos_1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'y vs x for {method} method for head-on collision with step size {step}')
        plt.show()
        plt.clf()
        plt.plot(times, x_pos_0)
        plt.plot(times, x_pos_1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'x vs t for {method} method for head-on collision with step size {step}')
        plt.show()
        plt.clf()
        plt.plot(times, x_vel_0)
        plt.plot(times, x_vel_1)
        plt.xlabel('t')
        plt.ylabel('v_x')
        plt.title(f'v_x vs t for {method} method for head-on collision with step size {step}')
        plt.show()
        plt.clf()
        final_vel = velocities[:,-1,:]
        x_mom_ratio = x_momentum_ratio(init_vel, final_vel, 2)
        kinetic_ratio = kinetic_energy_ratio(init_vel, final_vel, 2)
        print(f'The ratio of initial x-momentum to final x-momentum in the {method} method with step size {step} is {x_mom_ratio}')
        print(f'The ratio of initial kinetic energy to final kinetic energy in the {method} method with step size {step} is {kinetic_ratio}')


init_pos = [[-5,1,0], [0,0,0]]
init_vel = [[0.2,0,0],[0,0,0]]

positions, velocities = simulate(init_pos, init_vel, 0.01, 2, 40, 'rk4')
x_pos_0 = positions[0,:,0]
y_pos_0 = positions[0,:,1]
x_pos_1 = positions[1,:,0]
y_pos_1 = positions[1,:,1]
x_vel_0 = velocities[0,:,0]
y_vel_0 = velocities[0,:,1]
x_vel_1 = velocities[1,:,0]
y_vel_1 = velocities[1,:,1]
times = np.arange(0, 40, 0.01)
plt.plot(x_pos_0, y_pos_0)
plt.plot(x_pos_1, y_pos_1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y vs x for rk4 method for offset collision with step size 0.01')
plt.show()
plt.clf()
plt.plot(times, x_pos_0)
plt.plot(times, x_pos_1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('x vs t for rk4 method for offset collision with step size 0.01')
plt.show()
plt.clf()
plt.plot(times, x_vel_0)
plt.plot(times, x_vel_1)
plt.xlabel('t')
plt.ylabel('v_x')
plt.title('v_x vs t for rk4 method for offset collision with step size 0.01')
plt.show()
plt.clf()
plt.plot(times, y_vel_0)
plt.plot(times, y_vel_1)
plt.xlabel('t')
plt.ylabel('v_y')
plt.title('v_y vs t for rk4 method for offset collision with step size 0.01')
plt.show()
plt.clf()

init_pos = [[-5,2,0], [0,0,0]]
init_vel = [[0.2,0,0],[0,0,0]]

positions, velocities = simulate(init_pos, init_vel, 0.01, 2, 40, 'rk4')
x_pos_0 = positions[0,:,0]
y_pos_0 = positions[0,:,1]
x_pos_1 = positions[1,:,0]
y_pos_1 = positions[1,:,1]
x_vel_0 = velocities[0,:,0]
y_vel_0 = velocities[0,:,1]
x_vel_1 = velocities[1,:,0]
y_vel_1 = velocities[1,:,1]
times = np.arange(0, 40, 0.01)
plt.plot(x_pos_0, y_pos_0)
plt.plot(x_pos_1, y_pos_1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y vs x for rk4 method for offset collision with step size 0.01')
plt.show()
plt.clf()
plt.plot(times, x_pos_0)
plt.plot(times, x_pos_1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('x vs t for rk4 method for offset collision with step size 0.01')
plt.show()
plt.clf()
plt.plot(times, x_vel_0)
plt.plot(times, x_vel_1)
plt.xlabel('t')
plt.ylabel('v_x')
plt.title('v_x vs t for rk4 method for offset collision with step size 0.01')
plt.show()
plt.clf()
plt.plot(times, y_vel_0)
plt.plot(times, y_vel_1)
plt.xlabel('t')
plt.ylabel('v_y')
plt.title('v_y vs t for rk4 method for offset collision with step size 0.01')
plt.show()
plt.clf()

l = 2 ** (1/6)
cos30 = (3 ** (1/2)) / 2
sin30 = 1/2

balls = [[-5, 0, 0], [0,0,0],[l*cos30, l*sin30, 0], [l*cos30, -1 * l * sin30,0],
            [2 * l*cos30, 2 * l*sin30, 0], [2 * l*cos30, 0, 0], [2 * l*cos30, -2 * l*sin30, 0],
            [3 * l*cos30, 3 * l*sin30, 0], [3 * l*cos30, l*sin30, 0], [3 * l*cos30, -1* l*sin30, 0], [3 * l*cos30, -3 * l*sin30, 0],
            [4 * l*cos30, 4 * l*sin30, 0], [4 * l*cos30, 2*l*sin30, 0], [4 * l * cos30, 0, 0], [4 * l*cos30, -2* l*sin30, 0], [4 * l*cos30, -4 * l*sin30, 0]]

#Just demonstrating what the initial layout looks like
init_pos_x = [ball[0] for ball in balls]
init_pos_y = [ball[1] for ball in balls]
plt.scatter(init_pos_x, init_pos_y)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()
plt.clf()

#Creating the initial velocity list
init_velocity_rest = [[0,0,0] for i in range(15)]
init_velocity_1 = [[0.2, 0, 0]] + init_velocity_rest

positions, velocities = simulate(balls, init_velocity_1, 0.1, 16, 50, 'rk4')
for i in range(16):
    x_pos_i = positions[i,:,0]
    y_pos_i = positions[i,:,1]
    plt.plot(x_pos_i, y_pos_i)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Argon Pool Break v_x = 0.2')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()
plt.clf()

init_velocity_2 = [[2, 0, 0]] + init_velocity_rest

positions, velocities = simulate(balls, init_velocity_2, 0.1, 16, 50, 'rk4')
for i in range(16):
    x_pos_i = positions[i,:,0]
    y_pos_i = positions[i,:,1]
    plt.plot(x_pos_i, y_pos_i)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Argon Pool Break v_x = 2')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()
plt.clf()

init_velocity_3 = [[5, 0, 0]] + init_velocity_rest

positions, velocities = simulate(balls, init_velocity_3, 0.1, 16, 50, 'rk4')
for i in range(16):
    x_pos_i = positions[i,:,0]
    y_pos_i = positions[i,:,1]
    plt.plot(x_pos_i, y_pos_i)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Argon Pool Break v_x = 5')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()
plt.clf()
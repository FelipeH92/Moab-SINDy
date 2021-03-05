import csv
import numpy as np

def read_batch_moab(path, batch_number, reduced = False, dt = 1e-3):

    path_final = path + "/Batch_" + str(batch_number)
    # Posição
    x_pos = read_csv(path_final + "/ball_x.csv")
    y_pos = read_csv(path_final + "/ball_y.csv")
    x_batch = np.zeros((x_pos.shape[0],2))
    x_batch[:,0] = x_pos[:,0]
    x_batch[:,1] = y_pos[:,0]
    # Velocidade
    x_vel = read_csv(path_final + "/ball_vel_x.csv")
    y_vel = read_csv(path_final + "/ball_vel_y.csv")
    dx_batch = np.zeros((x_vel.shape[0],2))
    dx_batch[:,0] = x_vel[:,0]
    dx_batch[:,1] = y_vel[:,0]
    # Tempo
    t_batch = read_csv(path_final + "/time.csv")
    t_batch = t_batch.flatten()
    # Entrada
    theta_x = read_csv(path_final + "/theta_x.csv")
    theta_y = read_csv(path_final + "/theta_y.csv")
    u_batch = np.zeros((theta_x.shape[0],2))
    u_batch[:,0] = theta_x[:,0]
    u_batch[:,1] = theta_y[:,0]

    if (reduced == False):
        return x_batch, t_batch, u_batch, dx_batch
    
    x_batch, t_batch, u_batch, dx_batch = reduce_batch(x_batch, t_batch, u_batch, dx_batch, dt)
    return x_batch, t_batch, u_batch, dx_batch

def reduce_batch(x_batch, t_batch, u_batch, dx_batch, dt):
    t_list_aux = []
    x_pos = []
    y_pos = []
    x_vel = []
    y_vel = []
    theta_x = []
    theta_y = []
    count = 0.0
    for i in range(0,t_batch.shape[0]):
        if (t_batch[i] >= count):
            count = count+dt
            t_list_aux.append(t_batch[i])
            x_pos.append(x_batch[i,0])
            y_pos.append(x_batch[i,1])
            x_vel.append(dx_batch[i,0])
            y_vel.append(dx_batch[i,1])
            theta_x.append(u_batch[i,0])
            theta_y.append(u_batch[i,1])
    t_list_aux = np.asarray(t_list_aux, dtype=np.float128)
    t_list_aux = t_list_aux[:, np.newaxis]
    t_batch = t_list_aux.flatten()
    x_pos = np.asarray(x_pos, dtype=np.float128)
    x_pos = x_pos[:,np.newaxis]
    y_pos = np.asarray(y_pos, dtype=np.float128)
    y_pos = y_pos[:,np.newaxis]
    x_vel = np.asarray(x_vel, dtype=np.float128)
    x_vel = x_vel[:,np.newaxis]
    y_vel = np.asarray(y_vel, dtype=np.float128)
    y_vel = y_vel[:,np.newaxis]
    theta_x = np.asarray(theta_x, dtype=np.float128)
    theta_x = theta_x[:,np.newaxis]
    theta_y = np.asarray(theta_y, dtype=np.float128)
    theta_y = theta_y[:,np.newaxis]
    
    x_batch = np.zeros((t_batch.shape[0],2))
    dx_batch = np.zeros((t_batch.shape[0],2))
    u_batch = np.zeros((t_batch.shape[0],2))

    x_batch[:,0] = x_pos[:,0]
    x_batch[:,1] = y_pos[:,0]
    dx_batch[:,0] = x_vel[:,0]
    dx_batch[:,1] = y_vel[:,0]
    u_batch[:,0] = theta_x[:,0]
    u_batch[:,1] = theta_y[:,0]

    return x_batch, t_batch, u_batch, dx_batch

def read_csv(csv_path):
    data = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            data.append(row[0])
        data_array = np.asarray(data, dtype=np.float128)
        data_array = data_array[:,np.newaxis]
    csv_file.close()
    return data_array

'''
data_test = read_csv("/home/nascimento/Projects/Doutorado/SINDy/Code/SINDy_MOAB/src/data/Output_Sample_Time_1e4/Batch_1/ball_x.csv")
x_batch, t_batch, u_batch, dx_batch = read_batch_moab("/home/nascimento/Projects/Doutorado/SINDy/Code/SINDy_MOAB/src/data/Output_Sample_Time_1e4", 1)

for i in range(1, t_batch.shape[0]):
    if (t_batch[i,0] <= t_batch[i-1,0]):
        print("Strange time value in line ", i)
        print(t_batch[i-1,0])
        print(t_batch[i,0])

print(x_batch.shape)
'''
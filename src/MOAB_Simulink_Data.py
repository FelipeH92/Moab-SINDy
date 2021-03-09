import os
import numpy as np
import utils.data_manager as dm
import pysindy as ps

# O estado (posição e velocidade em x e y) e a entrada (angulo x e y do plate)
feature_names = ['x', 'dx', 'y', 'dy', 'theta_x', 'theta_y']
output_names = ['dx', 'ddx', 'dy', 'ddy']

def fit_moab_sindy(x_0, t_train, x_0_test, t_test, u_train = None, u_test = None, otimizador = ps.STLSQ(threshold=0.5, alpha=.05)):

    # Definindo a biblioteca de funções como funções polinomiais de grau 5
    library = ps.PolynomialLibrary(degree=5, include_interaction=False)

    # Otimizador
    optimizer = otimizador

    # Utilizando o SINDy
    model = ps.SINDy(feature_names=feature_names, feature_library=library, optimizer=optimizer)
    model.fit(x_0, t=t_train, u=u_train)
    model.print(lhs=output_names)

    print('Model score): %f' %model.score(x_0_test, u=u_test, t=t_test))
    print('\n')

    return model

# Simulação padrão, com passo de 0.045

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = dir_path + "/data/Output_Standard_Simulation"

for i in range(1,10):

    x_train, t_train, u_train, dx_train = dm.read_batch_moab(data_path, i, reduced = True)

    x_train_final = np.zeros((x_train.shape[0],4))
    x_train_final[:,0] = x_train[:,0]
    x_train_final[:,2] = x_train[:,1]
    x_train_final[:,1] = dx_train[:,0]
    x_train_final[:,3] = dx_train[:,1]

    x_test, t_test, u_test, dx_test = dm.read_batch_moab(data_path, i+1, reduced = True)

    x_test_final = np.zeros((x_test.shape[0],4))
    x_test_final[:,0] = x_test[:,0]
    x_test_final[:,2] = x_test[:,1]
    x_test_final[:,1] = dx_test[:,0]
    x_test_final[:,3] = dx_test[:,1]

    fit_moab_sindy(x_train_final,t_train,x_test_final,t_test,u_train=u_train, u_test=u_test)

# Simulação com passo 1e-3

print("Simulações com passo 1e-3")

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = dir_path + "/data/Output_Sample_Time_1e4"

for i in range(1,10):

    x_train, t_train, u_train, dx_train = dm.read_batch_moab(data_path, i, reduced = True)

    x_train_final = np.zeros((x_train.shape[0],4))
    x_train_final[:,0] = x_train[:,0]
    x_train_final[:,2] = x_train[:,1]
    x_train_final[:,1] = dx_train[:,0]
    x_train_final[:,3] = dx_train[:,1]

    x_test, t_test, u_test, dx_test = dm.read_batch_moab(data_path, i+1, reduced = True)

    x_test_final = np.zeros((x_test.shape[0],4))
    x_test_final[:,0] = x_test[:,0]
    x_test_final[:,2] = x_test[:,1]
    x_test_final[:,1] = dx_test[:,0]
    x_test_final[:,3] = dx_test[:,1]

    fit_moab_sindy(x_train_final,t_train,x_test_final,t_test,u_train=u_train, u_test=u_test)


# Melhor conjunto de dados

print("O Batch 8 do conjunto teve o melhor resultado nos dados.")
print('\n')

x_train, t_train, u_train, dx_train = dm.read_batch_moab(data_path, 8, reduced = True)

x_train_final = np.zeros((x_train.shape[0],4))
x_train_final[:,0] = x_train[:,0]
x_train_final[:,2] = x_train[:,1]
x_train_final[:,1] = dx_train[:,0]
x_train_final[:,3] = dx_train[:,1]

x_test, t_test, u_test, dx_test = dm.read_batch_moab(data_path, 6, reduced = True)

x_test_final = np.zeros((x_test.shape[0],4))
x_test_final[:,0] = x_test[:,0]
x_test_final[:,2] = x_test[:,1]
x_test_final[:,1] = dx_test[:,0]
x_test_final[:,3] = dx_test[:,1]

fit_moab_sindy(x_train_final,t_train,x_test_final,t_test,u_train=u_train, u_test=u_test)


# Simulação com passo pequeno e posição inicial definida

print("Simulações com passo 1e-3 e posição inicial definida")

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = dir_path + "/data/Output_Ramp"

for i in range(1,4):

    x_train, t_train, u_train, dx_train = dm.read_batch_moab(data_path, i, reduced = True)

    x_train_final = np.zeros((x_train.shape[0],4))
    x_train_final[:,0] = x_train[:,0]
    x_train_final[:,2] = x_train[:,1]
    x_train_final[:,1] = dx_train[:,0]
    x_train_final[:,3] = dx_train[:,1]

    x_test, t_test, u_test, dx_test = dm.read_batch_moab(data_path, i+1, reduced = True)

    x_test_final = np.zeros((x_test.shape[0],4))
    x_test_final[:,0] = x_test[:,0]
    x_test_final[:,2] = x_test[:,1]
    x_test_final[:,1] = dx_test[:,0]
    x_test_final[:,3] = dx_test[:,1]

    fit_moab_sindy(x_train_final,t_train,x_test_final,t_test,u_train=u_train, u_test=u_test)

x_train, t_train, u_train, dx_train = dm.read_batch_moab(data_path, 4, reduced = True)

x_train_final = np.zeros((x_train.shape[0],4))
x_train_final[:,0] = x_train[:,0]
x_train_final[:,2] = x_train[:,1]
x_train_final[:,1] = dx_train[:,0]
x_train_final[:,3] = dx_train[:,1]

x_test, t_test, u_test, dx_test = dm.read_batch_moab(data_path, 1, reduced = True)

x_test_final = np.zeros((x_test.shape[0],4))
x_test_final[:,0] = x_test[:,0]
x_test_final[:,2] = x_test[:,1]
x_test_final[:,1] = dx_test[:,0]
x_test_final[:,3] = dx_test[:,1]

fit_moab_sindy(x_train_final,t_train,x_test_final,t_test,u_train=u_train, u_test=u_test, otimizador = ps.STLSQ(threshold=0.5, alpha=.05))
import os
import numpy as np
import utils.data_manager as dm
import pysindy as ps

# O estado (posição e velocidade em x e y) e a entrada (angulo x e y do plate)
feature_names = ['x', 'y']
output_names = ['dx', 'dy']

def fit_moab_sindy(x_0, t_train, x_0_test, t_test, u_train = None, u_test = None, otimizador = ps.STLSQ(threshold=0.2)):

    # Definindo a biblioteca de funções como funções polinomiais de grau 5
    library = ps.PolynomialLibrary(degree=5)

    # Otimizador
    optimizer = otimizador

    # Utilizando o SINDy
    model = ps.SINDy(feature_names=feature_names, feature_library=library, optimizer=optimizer)
    model.fit(x_0, t=t_train, u=u_train)
    model.print(lhs=output_names)

    print('Model score): %f' %model.score(x_0_test, u=u_test, t=t_test))
    print('\n')

    return model

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = dir_path + "/data/Output_Sample_Time_1e4"

x_train, t_train, _, _ = dm.read_batch_moab(data_path, 1)

x_test, t_test, _, _ = dm.read_batch_moab(data_path, 2)

fit_moab_sindy(x_train,t_train,x_test,t_test)
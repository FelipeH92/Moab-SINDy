import numpy as np
from scipy.integrate import odeint
import utils.moab_models as moab
import pysindy as ps

# O estado (posição e velocidade em x e y) e a entrada (angulo x e y do plate)
feature_names = ['x', 'dx', 'y', 'dy', 'thetax', 'dthetax', 'thetay', 'dthetay']
output_names = ['dx', 'ddx', 'dy','ddy']

def fit_moab_sindy(x_0, t_train, x_0_test, t_test, Amplitude = 2, otimizador = ps.STLSQ(threshold=0.2)):
    #Amplitude do seno do angulo de entrada
    moab.A = Amplitude

    # Calculando o estado X. Usando odeint com o moab_simplificado, a entrada será a velocidade da bola.
    x_train = odeint(moab.moab, x_0, t_train, args=(moab.sin6tpidt, moab.sin6tpi2dt))

    u_train = np.zeros((t_train.shape[0],4))
    u_train[:,0:2] = moab.sin6tpidt(t_train,stack=True)
    u_train[:,2:4] = moab.sin6tpi2dt(t_train,stack=True)

    # Definindo a biblioteca de funções como funções polinomiais de grau 5
    library = ps.PolynomialLibrary(degree=5)

    # Otimizador
    optimizer = otimizador

    # Utilizando o SINDy
    model = ps.SINDy(feature_names=feature_names, feature_library=library, optimizer=optimizer)
    model.fit(x_train, u=u_train, t=dt)
    model.print(lhs=output_names)

    # Criando uma base de testes
    # t_test = np.arange(0, 4.5, dt) # 1.5 segundos a mais
    # x0_test = np.array([0.08, 0, 0.08, 0]) # Condição inicial com a bola em 8 centimetros em x e y
    x_test = odeint(moab.moab, x_0_test, t_test, args=(moab.sin6tpidt, moab.sin6tpi2dt))

    u_test = np.zeros((t_test.shape[0],4))
    u_test[:,0:2] = moab.sin6tpidt(t_test,stack=True)
    u_test[:,2:4] = moab.sin6tpi2dt(t_test,stack=True)

    print('Model score (Amplitude ', Amplitude, '): %f' %model.score(x_test, u=u_test, t=dt))
    print('\n')

    return model

# Passo de cálculo
dt = 0.001
moab.dt = dt

# Intervalo de cálculo
t_train = np.arange(0,2,dt)

# Condição Inicial
x_0 = [0, 0, 0, 0]

# Condições de teste
t_test = np.arange(0, 4.5, dt) # 1.5 segundos a mais
x0_test = np.array([0.08, 0, 0.08, 0]) # Condição inicial com a bola em 8 centimetros em x e y

fit_moab_sindy(x_0,t_train,x0_test,t_test)

## Outros casos: Mudanças na Amplitude da entrada

fit_moab_sindy(x_0,t_train,x0_test,t_test, Amplitude=3)
fit_moab_sindy(x_0,t_train,x0_test,t_test, Amplitude=5)

# Outro otimizador e amplitudes diferentes
optimizer=ps.STLSQ(threshold=0.3)
print('Limiar do otimizador em 0.3')
print('\n')

fit_moab_sindy(x_0,t_train,x0_test,t_test, otimizador=optimizer)
fit_moab_sindy(x_0,t_train,x0_test,t_test, Amplitude=3, otimizador=optimizer)
fit_moab_sindy(x_0,t_train,x0_test,t_test, Amplitude=5, otimizador=optimizer)

# Caso mais extremo. Amplitude 10 graus e threshold do otimizador 0.4

print('Caso mais extremo. Amplitude 10 graus e threshold do otimizador 0.4')
print('\n')
optimizer = ps.STLSQ(threshold=0.4)

fit_moab_sindy(x_0,t_train,x0_test,t_test, Amplitude=10, otimizador=optimizer)
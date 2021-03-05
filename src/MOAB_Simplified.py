import numpy as np
from scipy.integrate import odeint
import utils.moab_models as moab
import pysindy as ps

# O estado (velocidade em x e y) e a entrada (angulo x e y do plate)
feature_names = ['dx', 'dy', 'thetax', 'thetay']
output_names = ['ddx', 'ddy']

def fit_moab_simplified_sindy(x_0,t_train,x_0_test,t_test, Amplitude=2, library = ps.PolynomialLibrary(degree=5, include_interaction=False), otimizador=None):

    # Amplitude do seno de entrada
    moab.A = Amplitude

    # Calculando o estado X. Usando odeint com o moab_simplificado, a entrada será a velocidade da bola.
    x_train = odeint(moab.moab_simplified, x_0, t_train, args=(moab.sin6tpi, moab.sin6tpi2))

    # Calculando a entrada usando dois senos.
    u_train = np.zeros((t_train.shape[0],2))
    u_train[:,0] = moab.sin6tpi(t_train,stack=True)
    u_train[:,1] = moab.sin6tpi2(t_train,stack=True)

    # Definindo a biblioteca de funções como funções polinomiais de grau 3, excluindo as interações entre variaveis
    library = library

    # Utilizando o SINDy
    if (otimizador == None):
        model = ps.SINDy(feature_names=feature_names, feature_library=library)
    else:
        model = ps.SINDy(feature_names=feature_names, feature_library=library, optimizer=otimizador)
    model.fit(x_train, u=u_train, t=dt)
    model.print(lhs=output_names)

    # Criando uma base de testes
    
    x_test = odeint(moab.moab_simplified, x0_test, t_test, args=(moab.sin6tpi, moab.sin6tpi2))

    u_test = np.zeros((t_test.shape[0],2))
    u_test[:,0] = moab.sin6tpi(t_test,stack=True)
    u_test[:,1] = moab.sin6tpi2(t_test,stack=True)

    print('Model score (Amplitude ', Amplitude, '): %f' % model.score(x_test, u=u_test, t=dt))
    print('\n')

    return model

# Passo de cálculo
dt = 0.001

# Intervalo de cálculo
t_train = np.arange(0,2,dt)

# Condição Inicial
x_0 = [0, 0]

t_test = np.arange(0, 4.5, dt) # 1.5 segundos a mais
x0_test = np.array([0.05, 0.05]) # Condição inicial com a bola em 5 centimetros em x e y

print("Teste Padrão: Sem otimizador especifico, amplitude 2, sem interação entre variáveis")
print('\n')

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test)

# Outros testes
# Aumentando a amplitude dos angulos para 3 graus e para 5 graus ao invés de 2.

print("Teste: Sem otimizador especifico, amplitude 3 e 5, sem interação entre variáveis")
print('\n')

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=3)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=5)

# Os mesmos testes, incluindo interações entre variaveis dos polinômios

print("Teste: Sem otimizador especifico, amplitude 2, 3 e 5, com interação entre variáveis")
print('\n')

library = ps.PolynomialLibrary(degree=5, include_interaction=True)

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=2, library=library)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=3, library=library)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=5, library=library)

# Mesmos testes (sem interações), mudando o modo de otimização

print("Teste: Com otimizador STLSQ de limiar 0.1, amplitude 2, 3 e 5, sem interação entre variáveis")
print('\n')

optimizer = ps.STLSQ(threshold=0.1)
library = ps.PolynomialLibrary(degree=5, include_interaction=False)

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=2, library=library, otimizador=optimizer)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=3, library=library, otimizador=optimizer)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=5, library=library, otimizador=optimizer)

# Final, biblioteca com interações e otimizador

print("Teste: Com otimizador STLSQ de limiar 0.2, amplitude 2, 3 e 5, com interação entre variáveis")
print('\n')

optimizer = ps.STLSQ(threshold=0.2)
library = ps.PolynomialLibrary(degree=5, include_interaction=True)

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=2, library=library, otimizador=optimizer)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=3, library=library, otimizador=optimizer)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=5, library=library, otimizador=optimizer)
import numpy as np
from scipy.integrate import odeint
import utils.moab_models as moab
import pysindy as ps

# O estado (velocidade em x e y) e a entrada (angulo x e y do plate)
feature_names = ['dx', 'dy', 'thetax', 'thetay']
output_names = ['ddx', 'ddy']

def fit_moab_simplified_sindy(x_0,t_train,x_0_test,t_test, Amplitude=2, library = ps.PolynomialLibrary(degree=5, include_interaction=False), otimizador=None, u_x_function = moab.sin6tpi, u_y_function = moab.sin6tpi2):

    # Amplitude do seno de entrada
    moab.A = Amplitude

    # Calculando o estado X. Usando odeint com o moab_simplificado, a entrada será a velocidade da bola.
    x_train = odeint(moab.moab_simplified, x_0, t_train, args=(u_x_function, u_y_function))

    # Calculando a entrada usando dois senos.
    u_train = np.zeros((t_train.shape[0],2))
    u_train[:,0] = u_x_function(t_train,stack=True)
    u_train[:,1] = u_y_function(t_train,stack=True)

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
    
    x_test = odeint(moab.moab_simplified, x0_test, t_test, args=(u_x_function, u_y_function))

    u_test = np.zeros((t_test.shape[0],2))
    u_test[:,0] = u_x_function(t_test,stack=True)
    u_test[:,1] = u_y_function(t_test,stack=True)

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



####
# Invertendo o sinal de entrada
####

print("Teste Padrão Invertido: Sem otimizador especifico, amplitude 2, sem interação entre variáveis")
print('\n')

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)

# Outros testes
# Aumentando a amplitude dos angulos para 3 graus e para 5 graus ao invés de 2.

print("Teste Invertido: Sem otimizador especifico, amplitude 3 e 5, sem interação entre variáveis")
print('\n')

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=3,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=5,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)

# Os mesmos testes, incluindo interações entre variaveis dos polinômios

print("Teste Invertido: Sem otimizador especifico, amplitude 2, 3 e 5, com interação entre variáveis")
print('\n')

library = ps.PolynomialLibrary(degree=5, include_interaction=True)

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=2, library=library,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=3, library=library,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=5, library=library,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)

# Mesmos testes (sem interações), mudando o modo de otimização

print("Teste Invertido: Com otimizador STLSQ de limiar 0.1, amplitude 2, 3 e 5, sem interação entre variáveis")
print('\n')

optimizer = ps.STLSQ(threshold=0.1)
library = ps.PolynomialLibrary(degree=5, include_interaction=False)

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=2, library=library, otimizador=optimizer,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=3, library=library, otimizador=optimizer,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=5, library=library, otimizador=optimizer,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)

# Final, biblioteca com interações e otimizador

print("Teste Invertido: Com otimizador STLSQ de limiar 0.2, amplitude 2, 3 e 5, com interação entre variáveis")
print('\n')

optimizer = ps.STLSQ(threshold=0.2)
library = ps.PolynomialLibrary(degree=5, include_interaction=True)

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=2, library=library, otimizador=optimizer,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=3, library=library, otimizador=optimizer,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=5, library=library, otimizador=optimizer,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)



###
# Mudando a condição inicial para o modelo
###

# Condição Inicial
x_0 = [-0.05, 0.05]

print("Teste Padrão, condição inicial diferente: Sem otimizador especifico, amplitude 2, sem interação entre variáveis")
print('\n')

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test)

# Outros testes
# Aumentando a amplitude dos angulos para 3 graus e para 5 graus ao invés de 2.

print("Teste, condição inicial diferente: Sem otimizador especifico, amplitude 3 e 5, sem interação entre variáveis")
print('\n')

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=3)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=5)

# Os mesmos testes, incluindo interações entre variaveis dos polinômios

print("Teste, condição inicial diferente: Sem otimizador especifico, amplitude 2, 3 e 5, com interação entre variáveis")
print('\n')

library = ps.PolynomialLibrary(degree=5, include_interaction=True)

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=2, library=library)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=3, library=library)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=5, library=library)

# Mesmos testes (sem interações), mudando o modo de otimização

print("Teste, condição inicial diferente: Com otimizador STLSQ de limiar 0.1, amplitude 2, 3 e 5, sem interação entre variáveis")
print('\n')

optimizer = ps.STLSQ(threshold=0.1)
library = ps.PolynomialLibrary(degree=5, include_interaction=False)

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=2, library=library, otimizador=optimizer)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=3, library=library, otimizador=optimizer)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=5, library=library, otimizador=optimizer)

# Final, biblioteca com interações e otimizador

print("Teste, condição inicial diferente: Com otimizador STLSQ de limiar 0.2, amplitude 2, 3 e 5, com interação entre variáveis")
print('\n')

optimizer = ps.STLSQ(threshold=0.2)
library = ps.PolynomialLibrary(degree=5, include_interaction=True)

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=2, library=library, otimizador=optimizer)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=3, library=library, otimizador=optimizer)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=5, library=library, otimizador=optimizer)

###
# Mudando a condição inicial para o modelo
###

# Condição Inicial
x_0 = [0.05, 0.05]

print("Teste Padrão, condição inicial diferente: Sem otimizador especifico, amplitude 2, sem interação entre variáveis")
print('\n')

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)

# Outros testes
# Aumentando a amplitude dos angulos para 3 graus e para 5 graus ao invés de 2.

print("Teste, condição inicial diferente: Sem otimizador especifico, amplitude 3 e 5, sem interação entre variáveis")
print('\n')

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=3,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=5,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)

# Os mesmos testes, incluindo interações entre variaveis dos polinômios

print("Teste, condição inicial diferente: Sem otimizador especifico, amplitude 2, 3 e 5, com interação entre variáveis")
print('\n')

library = ps.PolynomialLibrary(degree=5, include_interaction=True)

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=2, library=library,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=3, library=library,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=5, library=library,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)

# Mesmos testes (sem interações), mudando o modo de otimização

print("Teste, condição inicial diferente: Com otimizador STLSQ de limiar 0.1, amplitude 2, 3 e 5, sem interação entre variáveis")
print('\n')

optimizer = ps.STLSQ(threshold=0.1)
library = ps.PolynomialLibrary(degree=5, include_interaction=False)

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=2, library=library, otimizador=optimizer,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=3, library=library, otimizador=optimizer,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=5, library=library, otimizador=optimizer,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)

# Final, biblioteca com interações e otimizador

print("Teste, condição inicial diferente: Com otimizador STLSQ de limiar 0.2, amplitude 2, 3 e 5, com interação entre variáveis")
print('\n')

optimizer = ps.STLSQ(threshold=0.2)
library = ps.PolynomialLibrary(degree=5, include_interaction=True)

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=2, library=library, otimizador=optimizer,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=3, library=library, otimizador=optimizer,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)
fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=5, library=library, otimizador=optimizer,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)


####
# Rampa na entrada
####

print("Rampa na Entrada")


# Intervalo de cálculo
t_train = np.arange(0,2,dt)

# Condição Inicial
x_0 = [0.05, -0.05]

optimizer = ps.STLSQ(threshold=0.2)
library = ps.PolynomialLibrary(degree=5, include_interaction=False)

moab.theta_ramp_x = 2*np.pi/180
moab.theta_ramp_y = 2*np.pi/180
moab.target = 0.5

moab.x_step = moab.theta_ramp_x/moab.target
moab.y_step = moab.theta_ramp_x/moab.target

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=2, library=library, otimizador=optimizer,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)


moab.theta_ramp_x = 3*np.pi/180
moab.theta_ramp_y = 3*np.pi/180
moab.target = 0.5

moab.x_step = moab.theta_ramp_x/moab.target
moab.y_step = moab.theta_ramp_x/moab.target

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=3, library=library, otimizador=optimizer,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)


moab.theta_ramp_x = 5*np.pi/180
moab.theta_ramp_y = 5*np.pi/180
moab.target = 0.5

moab.x_step = moab.theta_ramp_x/moab.target
moab.y_step = moab.theta_ramp_x/moab.target

fit_moab_simplified_sindy(x_0, t_train, x0_test, t_test, Amplitude=5, library=library, otimizador=optimizer,u_x_function=moab.sin6tpi2, u_y_function=moab.sin6tpi)

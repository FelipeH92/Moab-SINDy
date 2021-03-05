# Introdução

Esse repositório implementa o algoritmo SINDy Control com **PySINDy** para encontrar as equações diferenciais que definem o sistema _ball on plate_ **MOAB** da Microsoft.

Como estudo de caso do algoritmo SINDy, diferentes abordagens são realizadas:

1. Modelo simplificado do MOAB e sua aplicação com o SINDy.
2. Modelo padrão do MOAB e sua aplicação com o SINDy.
3. Uso apenas de dados da posição da bola (x e y) da simulação em Simulink (Deve dar errado).
4. Uso de dados da posição da bola (x e y) e da entrada da plataforma (\theta_x e \theta_y) n SINDy Control, e uso das velocidades da simulação.
5. Uso de dados de um sistema _ball on plate_ real.

# Dependencias necessários

São necessários as seguintes dependências instalados:

- NumPy
- Scypy
- Pysindy

# Organização de pastas

## src

Aqui cada estudo de caso do SINDy é computado.

## utils

Aqui estão scripts auxiliares como os modelos do MOAB, leitura e organização de arquivos CSV em Arrays, etc.

## data

Pasta com conjunto de dados de simulações em .csv para utilizar no algoritmo sindy. Cada vetor de dados se encontra em um csv separado dentro de uma pasta *Batch_#* de acordo com a seguinte estrutura:

- Posição x da bola: *ball_x.csv*
- Posição y da bola: *ball_y.csv*
- Velocidade x da bola: *ball_vel_x.csv*
- Velocidade y da bola: *ball_vel_y.csv*
- Ângulo theta x da plataforma: *theta_x.csv*
- Ângulo theta y da plataforma: *theta_y.csv*
- Tempo: *time.csv*
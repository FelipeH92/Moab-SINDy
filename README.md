# Introdução

Esse repositório implementa o algoritmo SINDy Control com **PySINDy** para encontrar as equações diferenciais que definem o sistema _ball on plate_ **MOAB** da Microsoft.

Como estudo de caso do algoritmo SINDy, diferentes abordagens são realizadas:

1. Modelo simplificado do MOAB e sua aplicação com o SINDy.
2. Modelo padrão do MOAB e sua aplicação com o SINDy.
3. Uso apenas de dados da posição da bola (x e y) da simulação em Simulink (Deve dar errado).
4. Uso de dados da posição da bola (x e y) e da entrada da plataforma (\theta_x e \theta_y) n SINDy Control.
5. Uso de dados da posição da bola (x e y) e da entrada da plataforma (\theta_x e \theta_y) n SINDy Control, e uso das velocidades da simulação.
6. Uso de dados de um sistema _ball on plate_ real.

# Pacotes necessários

São necessários os seguintes pacotes instalados:

- NumPy
- Scypy
- Pysindy

# Organização de pastas

## utils

Aqui estão scripts auxiliares como os modelos do MOAB, leitura e organização de arquivos CSV em Arrays, etc.

## src

Aqui cada estudo de caso do SINDy é computado.
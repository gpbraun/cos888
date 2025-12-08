# COS888

Otimização Combinatória II

## O projeto

Resolução de problemas de localização de facilidade capacitada utilizando os métodos:

1. Relax-and-Cut
2. Programação por Restrições
3. Decomposição de Benders
4. Geração de Colunas

## Experimentos SSCFL em Python

Instale o pacote local:

```shell
pip install -e sscfl
```

Para rodar os experimentos:

```shell
sscfl_experiments
```

Os resultados são registrados em: `out/sscfl_out.txt`.

Observação: para ativar a liceça do CPLEX no ambiente Python é necessário executar o comando:

```shell
docplex config --upgrade /opt/ibm/ILOG/CPLEX_Studio2212
```

## Experimentos TSCFL em C++

Primeiro faça o build:

```shell
cmake -S tscfl -B build
cmake --build build
```

Para rodar os experimentos:

```shell
build/tscfl_experiments
```

Os resultados são registrados em: `out/tscfl_out.txt`.

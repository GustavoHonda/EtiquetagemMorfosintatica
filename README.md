# EtiquetagemMorfosintatica
O objetivo deste exercício programa (EP) é exercitar os conhecimentos aprendidos em sala de aula desenvolvendo programas para etiquetagem morfossintática. No inglês, esta atividade se chama Part-of-Speech Tagging, ou simplesmente PoS tagging.

## Para testar 
export PYTHONPATH=$(pwd)
pytest

## Para rodar
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py

## Parte 1
Fiz todas as etapas de processamento sugeridas no enunciado, seleção das colunas, retirar as multiwords, inserção de tokens de início e fim; <s> e </s>, substitui palavras desconhecidas por <unk> e deixei todas as palavras em lower case.

Implementei uns arquivos de teste com pytest para monitorar se ainda estavam funcionando durante desenvolvimento.

Métricas Globais:
Acurácia: 0.15789473684210525, Precisão: 0.5263157894736842, Cobertura: 0.42105263157894735, Medida F1: 0.4678362573099415

## Parte2: 
Fiz a inicialização das matrizes com valores 1.

Tentei implementar o algoritmo de Baum Welch, mas fiquei muito confuso com todas as operações e tamanho das matrizes.
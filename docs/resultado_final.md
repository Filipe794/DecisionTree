O modelo **generalizou bem**: no conjunto de teste, todas as métricas ficaram iguais ou um pouco acima das médias da validação cruzada.

| métrica | CV (média ± desvio) | teste |
|---|---|---|
| ROC-AUC | 0,785 ± 0,049 | **0,790** |
| recall | 0,786 ± 0,120 | **0,802** |
| precisão | 0,543 ± 0,049 | 0,591 |
| F1 | 0,636 ± 0,056 | 0,681 |
| balanced_accuracy | 0,713 ± 0,046 | 0,751 |
| acurácia | 0,690 ± 0,046 | 0,736 |

Todas as métricas do teste ficam dentro de um desvio-padrão da média do CV. Não há sinal de que a escolha entre dezenas de configurações tenha sobreajustado o modelo à validação. O receio de que a AUC da melhor configuração fosse otimista não se confirmou para o modelo escolhido, provavelmente porque a regra "a mais simples entre as empatadas" evitou escolher a de maior média.

**A matriz de confusão, em termos clínicos.** São 231 pacientes de teste, 81 deles diabéticos:

| | previsto não diabético | previsto diabético |
|---|---|---|
| **não diabético (150)** | 105 | 45 |
| **diabético (81)** | 16 | 65 |

- **Encontra 80% dos diabéticos** (65 de 81). Deixa passar 16.
- **Dos 110 alarmes, 65 são diabéticos de fato** (precisão de 59%). Os outros 45 são alarmes falsos.
- **Quando diz "não diabético", acerta 87% das vezes** (105 de 121). Para triagem, esse é um número útil: um resultado negativo é bastante confiável.
- **Especificidade de 70%** (105 de 150 saudáveis classificados corretamente).
- Para comparação, prever "não diabético" para todos daria acurácia de 65% e recall 0. O modelo troca alguns acertos fáceis por encontrar a maioria dos doentes, como foi decidido no R3.

**Cuidados para o relatório:**
- É uma estimativa única, com 231 pacientes e só 81 diabéticos. A incerteza é grande: o intervalo de 95% do recall vai de aproximadamente 0,71 a 0,89. Os números do teste valem como confirmação de que o modelo generaliza, não como valores exatos.
- O teste foi usado uma vez. A partir daqui, nenhuma decisão do modelo pode mudar com base nesses números; caso contrário, o teste deixa de ser independente.

Preenchi o resultado do teste final na seção 2 das anotações. Os resultados também foram salvos em `results/final/20261024_145908_a1b1c4_R3_peso_balanceado`.

O que falta do plano é o R8 (robustez, opcional): repetir o modelo com outras sementes e outras divisões, numa cópia do projeto.
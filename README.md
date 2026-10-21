# Como usar o projeto

Árvore de decisão na base Pima Indians Diabetes, com um sistema de experimentos em que
cada método de tratamento é uma peça intercambiável e cada experimento é só uma
configuração no YAML. Nenhum resultado é sobrescrito: cada execução ganha uma pasta própria.

## 1. Instalação

```bash
pip install -r requirements.txt
```

No Colab, monte o Drive, entre na pasta do projeto e rode os mesmos comandos com `!`:

```python
%cd "/content/drive/MyDrive/Mestrado/Aprendizado de Máquina/DecisionTree"
!pip install -r requirements.txt
!python run_experiments.py experiments/experiments.yaml
```

Todos os comandos devem ser executados **a partir da raiz do projeto**.

## 2. Rotina de uso

```bash
# 1. gerar o split treino/teste (uma única vez; 70/30, estratificado, seed 42)
python -m src.make_split

# 2. editar experiments/experiments.yaml e executar
python run_experiments.py experiments/experiments.yaml

# 3. ver a comparação entre todas as execuções
python summarize.py                    # ordena por roc_auc
python summarize.py --metric recall    # ou por outra métrica

# 4. no fim, avaliar a(s) melhor(es) no teste
python final_test.py --run 20261021_142558_0e1f8f_E05_knn_winsor_smote
```

Opções do `run_experiments.py`:

| Opção | Efeito |
|---|---|
| `--only E02_knn E04_smote` | roda só esses experimentos |
| `--force` | roda de novo mesmo se já executado (cria pasta nova, não substitui) |
| `--n-jobs 1` | número de processos da validação cruzada (padrão: todos os núcleos) |

## 3. Estrutura de pastas

```
DecisionTree/
├── data/
│   ├── raw/pima-indians-diabetes.csv   nunca alterar
│   └── split/train.csv, test.csv       gerados por src.make_split
├── src/
│   ├── config.py      caminhos, SEED, CV (5 folds x 5 repetições), métricas
│   ├── data.py        load_train(), load_test()
│   ├── make_split.py  gera o split
│   ├── steps/         um arquivo por etapa de tratamento
│   ├── registry.py    catálogo: nome no YAML -> função que constrói o passo
│   ├── builder.py     config -> Pipeline
│   └── runner.py      validação cruzada + gravação dos resultados
├── experiments/experiments.yaml   a lista de experimentos
├── results/
│   ├── runs/          uma pasta por execução (só se acrescenta)
│   ├── final/         avaliações no teste
│   └── summary.csv    tabela gerada pelo summarize.py
├── run_experiments.py
├── summarize.py
└── final_test.py
```

No dia a dia só se mexe em dois lugares: `experiments/experiments.yaml` (para usar métodos)
e `src/steps/` + `src/registry.py` (para criar métodos novos).

## 4. O pipeline

Toda execução passa pelas mesmas etapas, sempre nesta ordem. Cada etapa recebe uma opção,
e `none` pula a etapa.

| # | Etapa | Opções | Observação |
|---|---|---|---|
| 1 | `zeros` | `none`, `to_nan` | 0 em Glucose, BloodPressure, SkinThickness, Insulin e BMI vira NaN |
| 2 | `missing_flags` | `none`, `flags` | cria colunas `<col>_missing` (0/1) antes da imputação |
| 3 | `outliers` | `none`, `winsor_1_99`, `winsor_5_95`, `winsor`, `iqr_clip`, `iqr_drop` | limites calculados só no treino de cada fold; `iqr_drop` remove linhas (só no treino) |
| 4 | `imputer` | `none`, `median`, `mean`, `knn`, `knn5`, `iterative` | preenche os NaN |
| 5 | `features` | `none`, `bmi_age_bins`, `ratios` | cria colunas novas |
| 6 | `selection` | `none`, `kbest`, `kbest8`, `rfe`, `rfe6` | escolhe colunas |
| 7 | `balancing` | `none`, `smote`, `random_over`, `random_under` | atua só no treino |
| 8 | `model` | hiperparâmetros do `DecisionTreeClassifier` | `random_state` já é fixado |

Parâmetros de cada opção (os valores mostrados são os padrões):

- `winsor`: `lower: 0.01`, `upper: 0.99`
- `iqr_clip` / `iqr_drop`: `k: 1.5`
- `knn`: `n_neighbors: 5`
- `iterative`: `max_iter: 10`
- `kbest`: `k: 8`, `score: f_classif` ou `mutual_info`
- `rfe`: `n_features: 6`
- `smote`: `k_neighbors: 5`

> Com as 8 colunas originais, `kbest8` mantém todas as colunas e não muda nada. Ele só tem
> efeito junto com `missing_flags` ou `features`.

## 5. Escrevendo experimentos

```yaml
baseline:
  zeros: to_nan
  missing_flags: none
  outliers: none
  imputer: median
  features: none
  selection: none
  balancing: none
  model: {max_depth: 4}

experiments:
  - name: E02_knn
    override: {imputer: {name: knn, n_neighbors: 5}}

  - name: E06_profundidade6
    override: {model: {max_depth: 6}}

  - name: E07_smote_raso
    override:
      balancing: {name: smote, k_neighbors: 3}
      model: {max_depth: 3, min_samples_leaf: 10}
```

Regras:

- O baseline roda automaticamente como `E01_baseline`.
- Cada etapa pode ser escrita como `nome` ou `{name: nome, parametro: valor}`.
- O `override` **substitui a etapa inteira**. Isso vale também para o `model`: para mudar só
  `min_samples_leaf` e manter a profundidade, repita `max_depth: 4`.
- Nomes de experimento não podem se repetir dentro do YAML.
- Antes de rodar qualquer coisa, o script valida todos os experimentos. Um erro de digitação
  para tudo com uma mensagem clara, por exemplo:
  `[E02_knn] imputer 'knnn' não existe, opções: none, median, mean, knn, knn5, iterative`.

Para tirar um experimento da lista, basta apagá-lo do YAML. As execuções antigas continuam
em `results/runs/`.

## 6. Onde ficam os resultados (e as árvores)

Cada execução cria uma pasta nova em `results/runs/`:

```
results/runs/20261021_142558_0e1f8f_E05_knn_winsor_smote/
├── config.yaml              config completa usada (baseline + override já resolvidos)
├── metrics.json             média e desvio de cada métrica na validação cruzada
├── fold_scores.csv          as 25 notas individuais (5 folds x 5 repetições)
├── meta.json                data, hash, versões das bibliotecas, hash do train.csv
├── pipeline.joblib          o pipeline treinado (tratamentos + árvore)
├── tree.txt                 a árvore em texto (regras)
├── tree.png                 a árvore desenhada
└── feature_importances.csv  importância de cada coluna na árvore
```

O nome da pasta é `data_hora` + `hash` + `nome`. O hash identifica o conteúdo (config +
seed + CV + métricas + arquivo de treino); o nome é só um rótulo.

### Qual árvore é salva

A validação cruzada treina **25 árvores**, uma por fold, só para medir o desempenho. Elas são
descartadas: o que fica delas são as notas em `fold_scores.csv` e `metrics.json`.

Depois disso, o runner treina **uma árvore final**, com o mesmo pipeline, usando o
`train.csv` inteiro. É essa árvore que fica salva na pasta:

- **`tree.txt`**: as regras da árvore, uma condição por linha. Cada folha mostra
  `weights: [negativos, positivos]` (quantas amostras de treino chegaram nela) e a classe prevista:
  ```
  |--- Insulin <= 123.11
  |   |--- Age <= 26.50
  |   |   |--- BMI <= 45.35
  |   |   |   |--- weights: [109.00, 2.00] class: 0
  ```
- **`tree.png`**: o desenho da árvore, com gini, número de amostras e classe em cada nó.
  Laranja indica a classe 0, azul a classe 1, e a cor mais forte indica um nó mais puro.
- **`feature_importances.csv`**: a importância de cada coluna, em ordem decrescente.
- **`pipeline.joblib`**: o objeto completo e treinado (imputador, winsorizador etc. + árvore),
  pronto para prever. Para usar num notebook:

  ```python
  import joblib
  from src.data import load_train

  pipe = joblib.load("results/runs/<pasta>/pipeline.joblib")
  arvore = pipe.named_steps["model"]        # o DecisionTreeClassifier
  print(arvore.get_depth(), arvore.get_n_leaves())

  X, y = load_train()
  pipe.predict(X.head())                    # aplica os tratamentos e a árvore
  ```

Observações:

- Os valores dos limiares estão na escala **depois** dos tratamentos. Com SMOTE, por exemplo,
  podem aparecer limiares quebrados como `Pregnancies <= 6.18`, porque o SMOTE cria amostras
  sintéticas interpoladas.
- O `.joblib` só abre com versões compatíveis do scikit-learn e do imbalanced-learn. As
  versões usadas ficam em `meta.json`.

### Tabela comparativa

`python summarize.py` lê todas as pastas de `results/runs/` e gera `results/summary.csv`,
ordenado pela métrica escolhida. A coluna `pipeline` resume o que cada execução usou, por exemplo
`to_nan > winsor_1_99 > knn(n_neighbors=5) > smote | max_depth=4`.

Essa tabela é sempre derivada das pastas: se algo der errado, apague-a e gere de novo.

### Garantias contra sobrescrita

- O runner só **cria** pastas. Ele nunca apaga nem edita nada em `results/runs/`.
- Se a mesma config, com os mesmos dados, já foi executada, ele avisa e pula. Com
  `--force`, cria uma pasta nova ao lado da antiga.
- Enquanto roda, a execução fica numa pasta `.tmp_...`. Ela só ganha o nome definitivo
  quando tudo foi gravado, então uma execução interrompida nunca conta como feita.
- Se você mudar o **código** de um método já existente (não a config), o hash não muda e o
  runner vai pular. Nesse caso, use `--force`.

## 7. Teste final

Só o `final_test.py` lê o `test.csv`. Ele:

1. carrega o `pipeline.joblib` da execução indicada, ou seja, **a mesma árvore** de `tree.txt`/`tree.png`;
2. avalia uma vez no teste;
3. grava `metrics.json`, `confusion_matrix.csv` e `meta.json` em `results/final/<run>/`.

Se a execução já foi avaliada, ele apenas mostra o resultado guardado, sem avaliar de novo.

## 8. Criando um método novo

Exemplo: um imputador que preenche com a moda.

1. Em `src/steps/imputers.py`:
   ```python
   def most_frequent():
       return SimpleImputer(strategy="most_frequent")
   ```
2. Em `src/registry.py`, na entrada `"imputer"`:
   ```python
   "most_frequent": imputers.most_frequent,
   ```
3. No YAML:
   ```yaml
   - name: E08_moda
     override: {imputer: most_frequent}
   ```

Transformadores que aprendem algo com os dados (quantis, médias...) devem calcular esses
valores no `fit` e só aplicá-los no `transform`, seguindo o padrão de `Winsorizer` em
`src/steps/outliers.py`. É isso que impede a contaminação com os dados de validação.

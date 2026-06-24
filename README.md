# Prevendo alto desempenho em Matemática no ENEM — Rio de Janeiro, 2022-2024

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vdms/mvp-machine-learning/blob/main/notebooks/mvp_ml_analytics_enem_rio.ipynb)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange)

Este repositório contém um MVP da disciplina de **Machine Learning & Analytics** de uma pós-graduação lato sensu. O projeto usa microdados públicos do ENEM para investigar uma pergunta central para a educação brasileira:

> Em que medida características socioeducacionais, escolares e de contexto antecipam alto desempenho em Matemática?

O foco não é construir um modelo para rotular estudantes. O valor do trabalho está em formular um problema de ML de forma responsável, controlar vazamento de dados, comparar modelos contra baselines honestos e interpretar os erros como evidência sobre desigualdade educacional.

Notebook principal: [`notebooks/mvp_ml_analytics_enem_rio.ipynb`](notebooks/mvp_ml_analytics_enem_rio.ipynb)

## Pergunta de ML

O problema foi formulado como uma **classificação binária supervisionada**:

```text
target = 1 se a nota de Matemática estiver no quartil superior
target = 0 caso contrário
```

Para evitar vazamento, o limiar de alto desempenho é calculado como o **percentil 75 da nota de Matemática apenas no conjunto de treino**. O modelo recebe somente variáveis disponíveis antes da avaliação ou independentes do resultado da prova:

- perfil do participante;
- faixa etária, sexo e cor/raça;
- trajetória escolar;
- características da escola;
- respostas do questionário socioeconômico;
- agregados municipais quando a granularidade individual não está disponível.

Notas, respostas, gabaritos, presença e qualquer variável diretamente ligada ao resultado da prova foram excluídos do conjunto de features.

## Dados e recorte

Os dados usados são os **Microdados do ENEM** publicados oficialmente pelo INEP, sem login, token ou API key.

| Item | Definição |
|---|---|
| Fonte | INEP — Microdados do ENEM |
| Período | 2022, 2023 e 2024 |
| Recorte geográfico | Estado do Rio de Janeiro (`SG_UF_PROVA == "RJ"`) |
| Elegibilidade | Presença na prova de Matemática e nota válida |
| Amostra final | 557.163 resultados elegíveis e 822.583 perfis RJ |

Os arquivos brutos do INEP são grandes e não são versionados. Em execução local, o notebook reaproveita bases derivadas em `outputs/` ou processa os ZIPs oficiais em `data/raw/`. No Google Colab, o notebook usa bases derivadas públicas dos microdados oficiais para evitar baixar e processar cerca de 1,8 GB durante a avaliação.

## Hipóteses

O MVP testa três hipóteses:

- **H1:** variáveis socioeducacionais e escolares possuem capacidade preditiva relevante para alto desempenho em Matemática.
- **H2:** parte dos padrões aprendidos em 2022-2023 permanece útil em 2024.
- **H3:** modelos supervisionados superam baselines ingênuos nos desenhos avaliados.

Essas hipóteses são tratadas como perguntas analíticas, não como prova causal.

## Desenho experimental

O projeto usa dois experimentos complementares. Eles respondem a perguntas diferentes e, por isso, não devem ser lidos como uma competição direta.

### Modelo A — pergunta socioeducacional individual

O Modelo A usa registros individuais de 2022 e 2023. O conjunto é dividido em treino e teste estratificados, mantendo o target em aproximadamente 25% de casos positivos. Este experimento mede quanto as informações socioeducacionais e escolares ajudam a prever alto desempenho dentro do mesmo período histórico.

### Modelo B — robustez temporal

O Modelo B treina em 2022-2023 e avalia em 2024. Como o INEP mudou o schema de 2024 e os perfis deixam de se ligar individualmente aos resultados, o experimento usa agregados municipais como contexto. Essa escolha reduz granularidade e cria risco de falácia ecológica, mas preserva uma pergunta importante: há sinal útil fora do período de treino?

O notebook também inclui uma decomposição da queda temporal para separar o efeito da mudança de ano do efeito da perda de granularidade individual.

## Pipeline de Machine Learning

O fluxo segue as etapas esperadas de um projeto de ML reprodutível:

1. carga dos microdados por ano;
2. filtro do recorte RJ e elegibilidade em Matemática;
3. definição do target sem usar o teste;
4. auditoria de features permitidas e excluídas;
5. análise exploratória;
6. `train_test_split` estratificado ou separação temporal, conforme o experimento;
7. imputação, codificação e escalonamento dentro de `Pipeline` e `ColumnTransformer`;
8. comparação contra baselines;
9. treinamento de `LogisticRegression`, `RandomForestClassifier` e Random Forest otimizada;
10. validação cruzada no treino;
11. avaliação final;
12. análise de erros, equidade e limitações.

Todo pré-processamento é ajustado somente no treino. O `GridSearchCV` também ocorre apenas dentro do conjunto de treino.

## Modelos e métricas

A métrica principal é o **F1-score**, porque a classe positiva representa aproximadamente o quartil superior de desempenho. Nesse contexto, accuracy isolada pode ser enganosa: um classificador conservador pode acertar muitos casos negativos e ainda falhar justamente em identificar estudantes de alto desempenho.

Modelos avaliados:

- `DummyClassifier` com estratégia majoritária;
- `DummyClassifier` estratificado, usado como baseline honesto para ganho de F1;
- `LogisticRegression`;
- `RandomForestClassifier`;
- `RandomForestClassifier` otimizada com `GridSearchCV`.

ROC-AUC, precision, recall, matrizes de confusão e análises por subgrupo complementam a avaliação.

## Resultados principais

| Experimento | Melhor modelo | F1 | ROC-AUC | Baseline estratificado (F1) | Ganho relativo |
|---|---|---:|---:|---:|---:|
| Modelo A — teste interno 2022-2023 | LogisticRegression | 0,584 | 0,813 | 0,250 | +133,5% |
| Modelo B — teste externo 2024 | LogisticRegression | 0,404 | 0,657 | 0,237 | +70,5% |

Principais achados:

- A `LogisticRegression` foi o melhor modelo nos dois desenhos, combinando desempenho competitivo, estabilidade e interpretabilidade.
- No Modelo A, a Random Forest simples teve F1 0,575 e a otimizada 0,574; a otimização não superou o modelo linear.
- No Modelo B, a Random Forest otimizada melhorou a Random Forest simples, mas ainda ficou abaixo da Logistic Regression em F1.
- O ganho sobre o baseline estratificado sustenta a hipótese de que há sinal preditivo nas variáveis socioeducacionais e escolares.
- Renda, tipo de escola e outros marcadores socioeducacionais aparecem como sinais fortes, mas isso deve ser lido como associação observacional, não causalidade.

## Interpretação dos erros e equidade

A parte mais importante do MVP não é apenas saber qual modelo venceu. É entender **como ele erra**.

No Modelo A, o modelo deixa de identificar uma parcela maior de estudantes de alto desempenho em alguns grupos:

- escola pública: taxa de falso negativo de aproximadamente 49%;
- escola privada: taxa de falso negativo de aproximadamente 11%;
- estudantes pretos: taxa de falso negativo de aproximadamente 60%;
- estudantes brancos: taxa de falso negativo de aproximadamente 16%.

Esses resultados mostram que a utilidade do modelo não se distribui de forma uniforme. Um modelo com bom desempenho médio pode falhar de maneira mais grave justamente nos grupos que políticas educacionais deveriam enxergar melhor.

Por isso, o projeto não recomenda uso para classificação individual de estudantes. A contribuição mais defensável é analítica: quantificar padrões, expor desigualdades e demonstrar um pipeline de ML tecnicamente responsável.

## Robustez temporal e schema de 2024

O teste externo em 2024 é mais difícil por duas razões:

1. muda o ano de avaliação;
2. muda a granularidade das features, pois perfis e resultados deixam de se ligar individualmente.

A decomposição implementada no notebook indica que a maior parte da queda de F1 vem da perda de granularidade individual, não apenas do passar do tempo.

O projeto também detecta uma mudança silenciosa no schema do INEP: em 2024, `Q006` deixa de representar a mesma pergunta de renda usada em anos anteriores. Quando as features agregadas de `Q006` são removidas no Modelo B, o F1 sobe de 0,404 para 0,423. Isso reforça uma lição importante de ML aplicado: mudanças de schema podem parecer drift temporal se não forem auditadas.

## Rastreabilidade

O notebook gera artefatos para sustentar as decisões e permitir auditoria:

- `artifacts/model_results.csv` e `.json`: métricas dos modelos;
- `artifacts/honest_baseline_gain.csv`: ganho contra baseline estratificado;
- `artifacts/feature_audit*.csv`: features permitidas, removidas e auditadas;
- `artifacts/preprocessing_decisions.csv`: decisões de pré-processamento;
- `artifacts/subgroup_error_model_a.csv`: análise de erro por grupo;
- `artifacts/model_b_sensitivity_no_q006.csv`: sensibilidade ao schema de 2024;
- `artifacts/data_flow_trace.csv`: rastreabilidade do fluxo de dados;
- matrizes de confusão, gráficos de EDA e coeficientes do modelo linear.

Esses artefatos ajudam a transformar o notebook em um relatório executável, e não apenas em uma sequência de células.

## Limitações e ética

Este MVP é um estudo observacional. Ele mostra associações entre contexto socioeducacional e alto desempenho, mas não estabelece causalidade.

Principais limitações:

- o recorte é apenas o estado do Rio de Janeiro;
- a generalização para outros estados ou anos exige nova validação;
- o Modelo B usa agregados municipais, sujeitos a falácia ecológica;
- a mudança de schema em 2024 afeta a comparabilidade temporal;
- as métricas médias escondem diferenças importantes entre grupos;
- o modelo não deve ser usado para decisões individuais sobre estudantes.

O uso responsável deste trabalho é analítico e educacional: entender sinais, vieses, desigualdades e riscos metodológicos em um problema real de classificação.

## Como executar

### Google Colab

Use o badge no topo do README e execute todas as células em ordem. No Colab, o notebook baixa bases derivadas públicas salvas em `outputs/`, produzidas a partir dos microdados oficiais do INEP.

### Local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/mvp_ml_analytics_enem_rio.ipynb
```

A execução completa pode levar algumas dezenas de minutos. Quando não houver cache em `outputs/`, o processamento local dos ZIPs oficiais é a etapa mais demorada.

## Estrutura do repositório

```text
.
├── notebooks/
│   └── mvp_ml_analytics_enem_rio.ipynb
├── outputs/
│   ├── enem_rj_results_2022_2024.csv.gz
│   ├── enem_rj_profiles_2022_2024.csv.gz
│   └── enem_rj_2022_2024.csv.gz
├── data/
│   ├── raw/
│   └── processed/
├── requirements.txt
└── README.md
```

`data/raw/` e `data/processed/` não versionam os microdados. Eles existem como pontos de entrada e saída local do notebook.

Ao executar o notebook, a pasta `artifacts/` é criada ou atualizada localmente com métricas, auditorias, tabelas e gráficos de suporte.

## Autoavaliação

O principal desafio deste MVP foi formular um problema de ML que fosse tecnicamente válido e socialmente honesto. Em um dataset educacional, é fácil obter alguma capacidade preditiva usando variáveis socioeconômicas; mais difícil é reconhecer o que esse desempenho significa e, principalmente, o que ele não autoriza concluir.

As decisões mais importantes do projeto foram:

- definir o target sem olhar para o teste;
- excluir qualquer variável que vazasse o resultado da prova;
- comparar modelos contra baselines apropriados;
- tratar 2024 como teste externo, mesmo com perda de granularidade;
- interpretar erros por grupo, não apenas métricas agregadas;
- preferir um modelo mais interpretável quando a complexidade adicional não trouxe ganho claro.

O resultado é um MVP de ML completo o suficiente para demonstrar classificação supervisionada, avaliação, robustez temporal e interpretabilidade, mas cauteloso o bastante para não vender uma previsão educacional como ferramenta de decisão individual.

## Nota final

Este projeto deve ser lido como um exercício aplicado de Machine Learning & Analytics: um notebook reprodutível, com dados públicos, anti-vazamento explícito, métricas rastreáveis e discussão crítica dos erros. A conclusão central não é que estudantes podem ser reduzidos a um score, mas que modelos podem revelar, com números, a persistência de desigualdades que já atravessam o sistema educacional.

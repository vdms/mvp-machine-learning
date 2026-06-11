# Prevendo alto desempenho em Matemática no ENEM — Rio de Janeiro, 2022–2024

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vdms/mvp-ml-analytics-enem-rio/blob/main/notebooks/mvp_ml_analytics_enem_rio.ipynb)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange)

MVP da disciplina de **Machine Learning & Analytics** (pós-graduação lato sensu). Investiga, com dados públicos, uma pergunta central para a educação brasileira: **o quanto as condições socioeconômicas de um estudante antecipam seu desempenho em Matemática no ENEM?** — e o que os erros do modelo revelam sobre desigualdade educacional.

📓 **Notebook:** [`notebooks/mvp_ml_analytics_enem_rio.ipynb`](notebooks/mvp_ml_analytics_enem_rio.ipynb) · 🌐 **HTML executado:** [`notebooks/mvp_ml_analytics_enem_rio.html`](notebooks/mvp_ml_analytics_enem_rio.html)

---

## O problema

Classificação binária supervisionada:

> `target = 1` se a nota de Matemática ≥ **percentil 75** (calculado *apenas no conjunto de treino*) · `target = 0` caso contrário

As features são exclusivamente características disponíveis **antes ou independentemente do resultado da prova**: perfil do participante (faixa etária, sexo, cor/raça, trajetória escolar), atributos da escola e o questionário socioeconômico (Q001–Q025: renda, escolaridade dos pais, bens domésticos, acesso à internet). Nenhuma nota, resposta, gabarito ou indicador de presença entra no modelo — a disciplina anti-vazamento é auditada em células próprias e registrada em `artifacts/`.

**Hipóteses testadas:**

- **H1** — Variáveis socioeducacionais e escolares possuem capacidade preditiva relevante.
- **H2** — Parte dos padrões aprendidos em 2022–2023 permanece estável em 2024.
- **H3** — Modelos supervisionados superam um baseline ingênuo nos dois desenhos.

## Os dados

| | |
|---|---|
| **Fonte** | [Microdados do ENEM — INEP](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem) (oficial, público, sem login) |
| **Edições** | 2022, 2023 e 2024 |
| **Recorte** | `SG_UF_PROVA == "RJ"` (estado do Rio de Janeiro) |
| **Elegibilidade** | Presença na prova de Matemática + nota válida |
| **Amostra final** | **557.163** resultados elegíveis (169.524 / 185.323 / 202.316 por ano) e **822.583** perfis RJ |

Os dados brutos (~2 GB por edição) **não são versionados**: o notebook baixa os ZIPs automaticamente das URLs públicas do INEP na primeira execução (ou detecta cópias locais em `data/raw/`), lê em *chunks* e aplica os filtros ainda durante a leitura.

## O que foi feito

O notebook percorre o fluxo completo de um projeto de ML, com narrativa em Markdown antes de cada etapa:

1. **Carga e recorte** — download/leitura dos microdados das 3 edições, filtro RJ, elegibilidade em Matemática, rastreabilidade dimensional em `data_flow_trace.csv`.
2. **Pré-processamento auditável** — decisões registradas em tabela (o quê, por quê, onde, mitigação de vazamento); imputação, one-hot encoding e escala **dentro de `Pipeline`/`ColumnTransformer`**, ajustados só no treino; features constantes no recorte detectadas e removidas programaticamente.
3. **EDA** — distribuição de notas por ano, prevalência do target, recortes descritivos por tipo de escola, cor/raça e renda; dicionário das variáveis do questionário socioeconômico.
4. **Dois experimentos complementares** (não competem entre si):
   - **Modelo A — pergunta socioeducacional individual:** treino/teste 80/20 estratificado em 2022–2023, com atributos individuais (44 features).
   - **Modelo B — robustez temporal:** treino em 2022–2023 e teste externo em **2024**. Como o INEP mudou o schema de 2024 (perfis e resultados não se ligam individualmente), o perfil entra como **agregados municipais** — limitação real assumida em vez de contornada.
5. **Modelagem e otimização** — baseline `DummyClassifier` (most_frequent **e** estratificado), `LogisticRegression`, `RandomForestClassifier` e RF otimizada via `GridSearchCV` (só no treino). Métrica principal: **F1-score** (classe positiva ≈ 25%); ROC-AUC complementar.
6. **Validação cruzada** — 5-fold no treino do Modelo A para dar margem de erro ao F1 e testar se a vitória da LogReg sobre a RF é real.
7. **Interpretabilidade** — coeficientes (log-odds) do modelo vencedor conectam a previsão à narrativa de desigualdade.
8. **Análise de equidade** — taxas de falso negativo/positivo por tipo de escola e cor/raça, com discussão dos vieses de seleção.
9. **Decomposição da queda temporal** — experimento extra que separa o efeito "perder granularidade individual" do efeito "mudar de ano", tornando H2 interpretável.
10. **Qualidade de dados em produção** — detecção de uma mudança silenciosa de schema do INEP em 2024 e experimento de sensibilidade quantificando seu impacto.

## Resultados

| Experimento | Melhor modelo | F1 | ROC-AUC | Baseline estratificado (F1) | Ganho |
|---|---|---:|---:|---:|---:|
| **Modelo A** (teste interno 2022–2023) | LogisticRegression | **0,584** | 0,813 | 0,250 | **+133%** |
| **Modelo B** (teste externo 2024) | LogisticRegression | **0,404** | 0,657 | 0,237 | **+70%** |

### Principais achados

- 🏫 **Renda e escola dominam a previsão.** Gradiente consistente de renda (`Q006`: faixa mais baixa −0,94 log-odds → mais alta +0,68) e forte efeito da dependência administrativa (escola federal +1,0; estadual −0,7).
- ⚖️ **O modelo erra de forma desigual.** Deixa de identificar **~49%** dos alunos de alto desempenho de escola pública (vs. ~11% da privada) e **~60%** dos estudantes pretos de alto desempenho (vs. ~16% dos brancos). Evidência quantitativa — não apenas prosa — de que a utilidade do modelo não se distribui uniformemente.
- 🧩 **A queda em 2024 é majoritariamente perda de granularidade, não instabilidade.** A decomposição mostra: **~79%** da queda de F1 vem da troca de features individuais por agregados municipais (mesmo período); só **~21%** é efeito temporal.
- 🔎 **Mudança de schema do INEP detectada nos dados.** Em 2024, `Q006` deixa de ser a pergunta de renda (17 faixas) e vira uma pergunta binária — corrompendo silenciosamente as features de renda no teste externo. Removê-las **melhora** o F1 de 2024 (0,404 → 0,423): parte do "drift temporal" era artefato de schema.
- 📊 **LogReg e RandomForest são estatisticamente indistinguíveis** (CV 5-fold: 0,584±0,004 vs 0,574±0,005). A escolha do modelo linear se sustenta por interpretabilidade — e a otimização de hiperparâmetros, reportada com honestidade, não superou o modelo simples.

## Como executar

**No Colab (recomendado para avaliação):** clique no badge no topo e execute todas as células — sem login, token ou upload. Download + execução completa: algumas dezenas de minutos.

**Local:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/mvp_ml_analytics_enem_rio.ipynb
```

Reprodutibilidade: `RANDOM_STATE = 42` em todas as etapas; `scripts/inspect_enem_file.py` inspeciona os arquivos brutos, se desejado.

## Estrutura do repositório

```
.
├── notebooks/    # Notebook final (129 células) e versão HTML executada
├── artifacts/    # Evidências geradas pelo notebook: métricas por modelo,
│                 # matrizes de confusão, coeficientes, análises de equidade,
│                 # auditorias de features/vazamento e rastreabilidade do fluxo
├── scripts/      # Utilitário de inspeção dos microdados brutos
├── data/raw/     # Microdados do INEP (não versionados; download automático)
└── requirements.txt
```

## Limitações e ética

Estudo **observacional**: associação não é causalidade — variáveis socioeconômicas serem preditivas não significa que causem o desempenho. Os agregados municipais do Modelo B estão sujeitos à **falácia ecológica** e são tratados só como contexto. O recorte é o RJ; generalizar para outros estados ou anos exige validação externa. E a conclusão mais importante da análise de equidade: **este modelo não deve ser usado para rotular estudantes ou tomar decisões individuais** — seus erros recaem desproporcionalmente sobre os grupos que políticas educacionais mais precisam alcançar. O valor do trabalho está em demonstrar o fluxo técnico com rigor anti-vazamento e em **quantificar** desigualdades educacionais com dados públicos. Discussão completa nas seções finais do notebook.

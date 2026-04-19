# REVIEW_CONTEXT

## Objetivo

Auditar, validar e corrigir este projeto de Data Science para apresentação em banca técnica.

O projeto resolve o desafio de **predição de churn em M+3** e também entrega a **parte não supervisionada** do case.

A meta desta revisão é:

1. aumentar a robustez metodológica;
2. corrigir inconsistências entre scripts, saídas e documento final;
3. melhorar a qualidade do código e a reprodutibilidade;
4. deixar a narrativa mais defensável diante da banca;
5. fazer alterações mínimas, corretas e justificadas, sem refazer o projeto inteiro.

---

## Escopo do projeto

Este repositório contém scripts para:

- download da base Kaggle;
- construção de features mensais;
- criação de lags e diffs;
- montagem da `model_table`;
- treino do modelo supervisionado de churn;
- curva de campanha;
- análise não supervisionada;
- avaliação de impacto de negócio;
- notebooks e documento final em PDF.

Arquivos principais esperados:

- `00_download_kaggle.py`
- `01_build_features.py`
- `01b_add_lag_features.py`
- `10_build_model_table.py`
- `11_train_churn_model.py`
- `11b_campaign_curve.py`
- `12_unsupervised_analysis.py`
- `13_business_impact.py`
- notebooks `.ipynb`
- PDF final da entrega

Arquivos auxiliares de auditoria disponíveis no projeto:

- `audit_summary.json`
- `label_trust_by_month.csv`
- `scores_test_sample.csv`
- `model_table_sample.csv`

---

## Contexto do case

O enunciado pede:

- criação de target de churn 3 meses à frente;
- feature engineering;
- feature selection;
- modelagem supervisionada;
- avaliação de negócio;
- análise não supervisionada complementar;
- justificativas metodológicas;
- código organizado para apresentação linear em banca.

O caso foi desenvolvido para rodar localmente no VS Code.

---

## Situação atual validada

A auditoria já confirmou que os números centrais do projeto estão coerentes com os artefatos gerados.

### Resultados confirmados

- `scores_test` com **2.987.795 linhas**
- churn base no teste de **0.0473934791**
- `model_table` com **17.026.533 linhas** e **32 colunas**
- faixa de meses da `model_table`: **201501 até 201611**
- meses suspeitos por `label_trust`:
  - `201501` até `201512`
  - `201610`
  - `201611`
- meses confiáveis efetivos:
  - `201601` até `201609`

### Curva Top-K confirmada

- Top 1%:
  - `k = 29877`
  - `recall = 0.0029236875`
  - `churn_rate = 0.0138568129`
- Top 3%:
  - `k = 89633`
  - `recall = 0.0132201523`
  - `churn_rate = 0.0208851651`
- Top 5%:
  - `k = 149389`
  - `recall = 0.0878801147`
  - `churn_rate = 0.0832993058`
- Top 10%:
  - `k = 298779`
  - `recall = 0.3344726769`
  - `churn_rate = 0.1585185036`
- Top 20%:
  - `k = 597559`
  - `recall = 0.7406533806`
  - `churn_rate = 0.1755107027`

---

## Diagnóstico técnico já identificado

### 1. Problema metodológico importante na população analítica

Há forte suspeita de que a população analítica nasce de `user_logs`, e depois recebe `left join` das demais fontes.

Isso pode enviesar a base para clientes com uso registrado, excluindo clientes ativos com pouco ou nenhum uso no mês, que são justamente relevantes para churn.

Este é o ponto metodológico mais crítico do projeto.

### 2. Curva de campanha com problema no topo extremo

O churn base do teste é aproximadamente **4,74%**, mas:

- Top 1% tem churn rate de **1,39%**
- Top 3% tem churn rate de **2,09%**

Ou seja, o ranking parece ruim nos percentis mais extremos e só passa a agregar valor a partir de Top 5%, ficando forte em Top 10% e Top 20%.

A narrativa executiva atual deve ser ajustada para refletir isso.

### 3. Feature selection está fraca como justificativa técnica

O projeto tem feature engineering, mas a etapa de feature selection não está explícita nem bem defendida.

As variáveis cadastrais parecem ter sido preparadas em uma etapa anterior, mas não chegam ao dataset final de modelagem. Isso precisa ser:

- confirmado no pipeline;
- tratado como decisão explícita;
- ou corrigido, se foi perda acidental.

### 4. Pipeline provavelmente não é idempotente

Os diretórios de saída de datasets particionados precisam ser verificados.

Há suspeita de que `features_lag/` e `model_table/` sejam gravados com `write_to_dataset` sem limpeza prévia da pasta de saída, o que pode causar duplicação de dados em reruns.

### 5. Inconsistência entre comentário e split real

Verificar se existe comentário dizendo algo como:

- teste com últimos 6 meses confiáveis
- validação com 3 meses anteriores

mas a execução real usa outra configuração, como 4 meses de teste e 2 de validação.

Corrigir comentários, documentação e qualquer texto inconsistente.

### 6. Parte não supervisionada possivelmente está forte demais na narrativa

A leitura dos clusters pode estar assertiva demais para o método usado.

Verificar:

- se a escolha do número de clusters foi justificada;
- se os “drivers” dos clusters estão sendo interpretados de forma excessiva;
- se há linguagem que soa causal ou definitiva demais para uma análise exploratória.

### 7. Resultado econômico precisa ser apresentado como cenário, não verdade operacional

A análise de negócio parece útil, mas depende de hipóteses fortes.

Verificar se o texto final deixa claro que:

- trata-se de análise de sensibilidade;
- o ROI depende fortemente da política de oferta;
- o cenário com 1 mês grátis parece promissor, mas não deve ser vendido como certeza operacional.

---

## Tarefas prioritárias

## Prioridade 1 — auditoria do pipeline

Ler os scripts e responder objetivamente:

1. a população analítica final nasce de `user_logs`?
2. as variáveis cadastrais são perdidas entre `features` e `model_table`?
3. os diretórios de saída são limpos antes de escrever datasets particionados?
4. o split temporal real está alinhado com comentários e PDF?
5. existe qualquer risco de leakage ou uso indevido de informação futura?

Registrar essas respostas em `REVIEW_NOTES.md`.

---

## Prioridade 2 — correções de código

Fazer correções mínimas e seguras.

### Corrigir, se confirmado:

- limpeza de diretórios de saída antes de `write_to_dataset`;
- inconsistências entre comentário e execução real;
- pequenos bugs, código morto, variáveis não usadas e trechos confusos;
- documentação interna dos scripts.

### Corrigir com cuidado extra:

- definição da população analítica, se estiver enviesada;
- passagem ou exclusão explícita de variáveis cadastrais;
- qualquer lógica de curva de campanha ou análise econômica.

Evitar refatorações desnecessárias.

---

## Prioridade 3 — narrativa técnica

Gerar ou atualizar um arquivo `REVIEW_NOTES.md` contendo:

### 1. Problemas encontrados
Lista objetiva dos problemas técnicos, metodológicos e narrativos.

### 2. Alterações realizadas
Quais arquivos foram alterados e por quê.

### 3. Limitações remanescentes
O que ainda continua como limitação legítima do case.

### 4. Sugestões para defesa na banca
Frases curtas e defensáveis para explicar:
- a validação temporal;
- a limitação da janela confiável;
- o comportamento da curva Top-K;
- a lógica da análise não supervisionada;
- as premissas econômicas.

---

## Regras de trabalho

1. Antes de editar qualquer arquivo, produzir um diagnóstico curto e claro.
2. Fazer alterações mínimas, corretas e justificadas.
3. Não inventar resultados novos sem evidência.
4. Não alterar números do PDF sem confirmar nos artefatos ou no código.
5. Se houver dúvida, preferir comentário ou nota explícita em vez de assumir.
6. Preservar a estrutura geral do projeto.
7. Priorizar legibilidade e apresentação em banca.

---

## Resultado esperado

Ao final, o projeto deve ficar em um estado melhor para banca, com:

- código mais consistente;
- narrativa mais honesta e mais forte;
- limitações explicitadas;
- menor risco de questionamentos metodológicos óbvios;
- documentação de revisão clara.

---

## Entregáveis esperados da revisão

1. diagnóstico inicial no chat ou no terminal;
2. correções aplicadas no código;
3. `REVIEW_NOTES.md`;
4. opcionalmente, pequenos ajustes em comentários e textos auxiliares;
5. opcionalmente, um resumo final das mudanças realizadas.

---

## Ponto de atenção final

Não tratar este projeto como inválido. A auditoria anterior mostrou que:

- os números centrais parecem coerentes;
- a estrutura geral do case é boa;
- a validação temporal tem mérito;
- há valor real no trabalho.

O objetivo desta revisão é transformar uma entrega “boa, mas vulnerável” em uma entrega “defensável e tecnicamente mais robusta”.
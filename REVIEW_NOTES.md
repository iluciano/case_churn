# REVIEW_NOTES

## 1. Problemas encontrados

- A população analítica nasce de `user_logs` em `01_build_features.py`, com `left join` das demais fontes depois. Isso tende a excluir clientes ativos sem uso registrado no mês, o que introduz viés metodológico justamente para churn.
- As variáveis cadastrais (`city`, `bd`, `gender`, `registered_via`) entram em `features.parquet`, mas não são propagadas para `features_lag/` nem para a `model_table`. A perda parece ser estrutural do pipeline atual, não um detalhe de treino.
- Os datasets particionados `features_lag/` e `model_table/` eram escritos sem limpeza prévia. Em reruns, o pipeline podia acumular arquivos antigos e duplicar dados.
- O split temporal real do modelo é `2` meses para validação e `4` para teste dentro da janela confiável. A narrativa da banca precisa refletir esse desenho real.
- A curva Top-K auditada mostra topo extremo fraco: Top 1% e Top 3% abaixo da base de churn; o ganho relevante aparece a partir de Top 5%, sobretudo em Top 10% e Top 20%.
- A parte não supervisionada é útil como segmentação exploratória, mas não sustenta linguagem causal ou excessivamente definitiva.
- A análise econômica depende de hipóteses fortes de aceitação, permanência e ARPU; portanto deve ser apresentada como sensibilidade, não como projeção operacional garantida.
- Há artefatos auxiliares no repositório gerados por versões diferentes dos scripts. Alguns CSVs não batem com a lógica atual do código nem entre si, o que enfraquece a reprodutibilidade da entrega.

## 2. Alterações feitas

- Atualizei `scripts/01b_add_lag_features.py` para limpar `data/processed/features_lag/` antes de regravar o dataset particionado. Isso torna o passo idempotente e evita acúmulo de partições antigas.
- Atualizei `scripts/10_build_model_table.py` para limpar `data/processed/model_table/` antes de regravar a saída. Isso reduz o risco de duplicação silenciosa em reruns.
- Ajustei a mensagem final de `scripts/11_train_churn_model.py` para alinhar a narrativa do modelo ao comportamento real da curva Top-K, sem superprometer precisão no topo extremo.
- Ajustei a saída textual de `scripts/12_unsupervised_analysis.py` para reforçar explicitamente que os clusters são exploratórios.
- Ajustei a saída textual de `scripts/13_business_impact.py` para reforçar explicitamente que o resultado econômico é uma análise de sensibilidade.
- Criei este `REVIEW_NOTES.md` para registrar diagnóstico, correções mínimas realizadas, limitações remanescentes e recomendações de defesa.

## 3. Limitações remanescentes

- A definição da população analítica continua baseada em uso (`user_logs`). Eu não corrigi isso nesta etapa porque mudar a base-mãe altera o desenho metodológico do case e exigiria regenerar o pipeline completo.
- As variáveis cadastrais continuam fora da `model_table`. Isso também não foi alterado agora porque exige uma decisão explícita: reincorporar essas variáveis ao modelo ou formalizar sua exclusão como escolha metodológica.
- Os artefatos já existentes em `reports/` continuam potencialmente desatualizados em relação aos scripts. Após as correções, o ideal é regenerar os outputs principais para deixar o repositório consistente.
- A análise não supervisionada continua limitada por amostragem e por uma interpretação baseada em médias e silhouette; ela é adequada como complemento exploratório, não como fundamento causal.
- A análise econômica continua simplificada: não modela heterogeneidade de aceitação, elasticidade por segmento, custo operacional da campanha nem risco de canibalização.

## 4. Sugestões para defesa na banca

- Validação temporal: “O modelo foi avaliado em esquema out-of-time, respeitando a ordem temporal e evitando validação aleatória que inflaria desempenho em série histórica.”
- Janela confiável: “Nem toda safra tinha observação futura confiável para rotular churn em M+3, então restringimos treino e avaliação aos meses com `label_trust` adequado.”
- Curva Top-K: “O modelo é mais útil para priorização em faixas operacionais como Top 5%, Top 10% e Top 20%; no topo extremo, a precisão não ficou forte o suficiente para uma campanha ultra restritiva.”
- População analítica: “A principal limitação metodológica do case é a população nascer de `user_logs`, o que pode sub-representar clientes ativos com baixo uso; reconhecemos isso explicitamente.”
- Feature selection: “Houve seleção prática via pipeline final de variáveis comportamentais e de pagamento, mas a exclusão das cadastrais precisa ser lida como decisão estrutural do dataset final, não como evidência de irrelevância intrínseca.”
- Não supervisionado: “Os clusters foram usados como ferramenta de leitura exploratória da base e heterogeneidade de risco, não como segmentação causal ou definitiva.”
- Impacto econômico: “O resultado financeiro deve ser lido como análise de sensibilidade sob premissas explícitas; ele serve para comparar políticas, não para prometer ROI garantido.”

# CLAUDE.md — TC5 / Datathon POSTECH MLET

Contexto e regras deste projeto. Leia antes de qualquer alteração.

---

## 1. O que é este projeto

Datathon da Fase 5. Plataforma de **experimentação adaptativa** que decide, em canais digitais,
qual próximo passo apresentar a cada cliente elegível — usando **multi-armed bandit** no lugar de
regra fixa ou teste A/B longo.

Enunciado oficial: `POSTECH - MLET - DATATHON (1).pdf`, na raiz do repositório.

Peso da avaliação: **30%** clareza do problema e impacto de negócio, **70%** validação técnica
(código organizado, modelo superando o baseline, uso de MLflow, demo funcionando).

Plano de execução por fases: `docs/PLANO.md`.

---

## 2. Decisões fechadas

Não reabrir sem discussão explícita com o usuário.

| Tema | Decisão |
|---|---|
| Base de dados | Kaggle `bank-marketing` (henriqueyamahata) — o `bank-additional-full.csv` do UCI |
| Formulação do problema | Braços derivados de **colunas reais de ação** do dataset (canal × janela de contato) |
| Avaliação | **A + C** — ambiente calibrado nos dados como track principal, replay/rejection sampling sobre o log real como validação |
| Serviço (Etapa 5) | FastAPI + Docker |
| Rastreio (Etapa 7) | MLflow local |
| Ofertas de produto sintéticas | **Descartado.** Entra só como parágrafo de roadmap no README |

Racional da formulação: colunas como `contact`, `month` e `day_of_week` são **decisões da campanha**,
não atributos do cliente. Como todos os braços aparecem no log histórico, `P(y \| contexto, braço)`
é estimável de dado observado para todo braço — o ambiente é calibrado, não inventado.

---

## 3. Restrições invioláveis

Vêm do enunciado. Violar qualquer uma custa nota.

- **`duration` é proibida.** Vazamento temporal, citada nominalmente no PDF. Nunca entra em feature set,
  nem no ambiente, nem na API.
- **Sem recompensa sintética.** Toda probabilidade tem que ser estimada a partir de linha observada.
  Se um braço não tem suporte no log, ele não existe.
- **Toda a documentação vai no `README.md`.** O PDF pede consolidação explícita e dispensa "múltiplos
  arquivos soltos de governança". `CLAUDE.md`, `CHANGELOG.md` e `docs/PLANO.md` são ferramentas de
  desenvolvimento, não entregáveis — tudo que a banca precisa ler mora no README.
- Sem dados reais de cliente, identificadores, patrimônio, renda, gênero ou raça.
- O README precisa cobrir: link da base, **base legal, finalidade, minimização, retenção** e
  **humano no loop** para decisões sensíveis.
- Preservar a referência ao Kaggle (fonte, versão, licença, colunas, target, limitações).

---

## 4. Convenções de código

- **Python 3.12.**
- **Identificadores, funções e docstrings em inglês. Prosa, README e notebooks em pt-BR.**
- `src/` é biblioteca: nenhum efeito colateral no import, nenhuma leitura de arquivo em nível de módulo.
  Os entrypoints são `train.py` e `api/app.py`.
- **Notebooks não contêm lógica.** Eles importam de `src/` e apresentam. Se um notebook tem uma função
  de negócio dentro, ela está no lugar errado.
- Toda política de bandit é uma classe com a mesma interface: `select(context) -> arm` e
  `update(context, arm, reward)`. Sem exceções — o runner e o replay dependem disso.
- Type hints em toda função pública.
- Sem estado global mutável. Sem `np.random` global: RNG é injetado.
- Dependência nova exige justificativa no `CHANGELOG.md`.

## 5. Reprodutibilidade

- `SEED` vive em `src/config.py` e é propagada explicitamente até as políticas.
- Todo experimento roda com **múltiplas seeds** e reporta média com intervalo de confiança.
  Curva de uma seed só não é resultado.
- **Experimento não logado no MLflow não aconteceu.** Params e métricas, sempre.
- Artefatos serializados em `models/`, acompanhados da metadata que os gerou.

## 6. Fluxo de trabalho

- Toda mudança relevante entra no `CHANGELOG.md`, formato Keep a Changelog, em pt-BR.
- Testes em `tests/`, rodando com pytest. Políticas, ambiente e replay exigem teste.
- `git commit` e `git push` somente quando o usuário pedir.

## 7. O que não fazer

- Não commitar `data/raw/`, `mlruns/`, `models/*.joblib` — ficam no `.gitignore`.
- Não adicionar framework pesado. `HistGradientBoostingClassifier` do scikit-learn cobre o que
  LightGBM ou XGBoost cobririam aqui, sem dependência extra na imagem Docker.
- Não inflar o README com dezenas de seções. A banca pontua clareza, não volume.
- Não criar arquivos de governança avulsos (ver seção 3).
- Não definir braço que não tenha suporte amostral mínimo no log — validar na EDA antes de fixar.

# Roteiro da demo — Etapa 8

Requisições prontas para gravar o vídeo. Todas as saídas abaixo foram capturadas da API rodando,
não escritas à mão: é o que você vai ver na tela.

Para rodar tudo em sequência, com pausas para narrar:

```bash
./scripts/demo.sh
```

O script sobe a API, espera o `/health`, e para a cada passo esperando **Enter**. Ao final derruba
o que subiu. Se preferir controlar na mão, os comandos estão todos abaixo.

---

## Antes de gravar

```bash
make setup      # se ainda não fez
make data       # baixa a base, confere o SHA-256
make train      # ~3 min — gera models/ e popula o MLflow
```

Confira que existe `models/environment.joblib`. Sem ele a API sobe mas responde `degraded`.

Deixe **duas abas** abertas antes de começar a gravar:

| Aba | Endereço | Quando aparece |
|---|---|---|
| Swagger | http://localhost:8000/docs | 3:30 em diante |
| MLflow | http://localhost:5000 | ~3:00 |

```bash
make api      # terminal 1
make mlflow   # terminal 2
```

---

## Divisão dos 5 minutos

| Tempo | O quê | Onde |
|---|---|---|
| 0:00–1:00 | O problema de negócio | Slide ou README |
| 1:00–2:00 | Base, braços e por que não sintetizamos recompensa | README |
| 2:00–3:00 | Resultados: adaptativo vs baseline, curvas de regret | README + figuras |
| 3:00–3:30 | MLflow com os 77 runs | Navegador |
| 3:30–5:00 | **A API rodando** | Swagger + terminal |

O trecho que importa é o último. Os quatro primeiros você narra; o quinto tem que funcionar ao vivo.

---

## A frase que abre

> "Um banco precisa decidir por qual canal e em que dia abordar cada cliente. Hoje isso é regra
> fixa. Nós trocamos por um sistema que aprende — e medimos quanto isso vale, inclusive quando não
> vale."

---

## Passo 1 — O serviço está de pé

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

```json
{
    "status": "ok",
    "model_loaded": true,
    "n_arms": 6,
    "arms": ["cellular|early", "cellular|late", "cellular|mid",
             "telephone|early", "telephone|late", "telephone|mid"],
    "model_version": "9511502630"
}
```

**Fale:** seis braços — dois canais vezes três janelas de contato. E a versão do artefato carregado,
que é o que amarra a resposta a um modelo específico.

---

## Passo 2 — Cliente A: estudante de 18 anos

```bash
curl -s -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"age":18,"job":"student","marital":"single","education":"high.school",
       "default":"no","housing":"no","loan":"no","campaign":1,"pdays":999,
       "previous":0,"poutcome":"nonexistent"}'
```

| Braço | Probabilidade |
|---|---:|
| **`cellular\|late`** | **44,85%** |
| `cellular\|early` | 44,85% |
| `cellular\|mid` | 39,92% |
| `telephone\|late` | 25,96% |
| `telephone\|early` | 23,19% |
| `telephone\|mid` | 10,14% |

**Fale:** a API não devolve só o vencedor, devolve o ranking inteiro. Estudante converte muito acima
da taxa-base de 11%, e celular domina — o pior braço vale um quarto do melhor.

---

## Passo 3 — Cliente B: técnico de 49 anos, já com 6 ligações

**É o momento mais forte da demo.** Mesma API, cliente diferente, e o ranking **vira**.

```bash
curl -s -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"age":49,"job":"technician","marital":"married","education":"professional.course",
       "default":"no","housing":"yes","loan":"no","campaign":6,"pdays":999,
       "previous":0,"poutcome":"nonexistent"}'
```

| Braço | Probabilidade | |
|---|---:|---|
| **`telephone\|early`** | **7,91%** | ⬅ telefone fixo assume a liderança |
| `cellular\|late` | 7,64% | |
| `cellular\|early` | 6,55% | |
| `cellular\|mid` | 6,17% | ⬅ o melhor braço *na média* caiu para quarto |
| `telephone\|late` | 5,23% | |
| `telephone\|mid` | 4,05% | |

**Fale:** para quem já levou seis ligações no celular, o telefone fixo deixa de ser inferior.
`cellular|mid`, que é o melhor braço na média da base, cai para quarto lugar. **É isso que uma
regra fixa nunca faria** — ela mandaria celular para os dois clientes.

---

## Passo 4 — Cliente C: já converteu antes

```bash
curl -s -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"age":35,"job":"management","marital":"married","education":"university.degree",
       "default":"no","housing":"no","loan":"no","campaign":1,"pdays":6,
       "previous":1,"poutcome":"success"}'
```

| Braço | Probabilidade |
|---|---:|
| **`telephone\|mid`** | **72,81%** |
| `telephone\|late` | 72,81% |
| `telephone\|early` | 72,81% |
| `cellular\|mid` | 72,81% |
| `cellular\|late` | 72,81% |
| `cellular\|early` | 63,52% |

> ⚠️ **Não passe direto por este.** A recomendação sai como `telephone|mid` — um braço de telefone
> fixo, para o melhor cliente da base. Sem explicação, parece erro grosseiro. **Explique antes que
> a banca pergunte.**

**Fale:** quem já assinou numa campanha anterior é o sinal mais forte da base — 72% contra 11% da
média. Mas repare: **cinco braços empatam em 72,81%**, e o `is_tie` marca isso.

Não é bug, é a calibração isotônica: ela é uma função em degraus, e clientes de alta propensão caem
todos no mesmo patamar. Quando cinco braços empatam, qual deles sai no topo é desempate numérico,
não preferência — e é exatamente por isso que a resposta traz o flag em vez de fingir convicção.

Para este cliente a decisão correta é **escolher pelo custo do canal**, não pelo modelo. Um sistema
que sempre finge ter opinião é pior que um que sabe quando não tem.

---

## Passo 5 — O contrato recusa lixo

```bash
curl -s -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"age":49,"job":"astronauta","marital":"married","education":"professional.course",
       "default":"no","housing":"yes","loan":"no","campaign":6,"pdays":999,
       "previous":0,"poutcome":"nonexistent"}' -w '\nHTTP %{http_code}\n'
```

```
campo: job
msg  : Input should be 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', ...
HTTP 422
```

**Fale:** categoria desconhecida é recusada na porta, nomeando os valores válidos. Sem isso o
encoder receberia o valor, devolveria um vetor de zeros, e a API responderia algo plausível a
partir de lixo — que é a pior falha possível, porque é silenciosa.

---

## Passo 6 — Exploração × explotação, ao vivo

Mesmo cliente, três chamadas. `seed` fixa o sorteio só para a demo ser reproduzível.

```bash
BODY='{"age":18,"job":"student","marital":"single","education":"high.school",
       "default":"no","housing":"no","loan":"no","campaign":1,"pdays":999,
       "previous":0,"poutcome":"nonexistent"}'

for s in 3 53 65; do
  curl -s -X POST "http://localhost:8000/recommend?explore=true&seed=$s" \
    -H 'Content-Type: application/json' -d "$BODY"
done
```

| | Braço | Probabilidade | `explored` |
|---|---|---:|---|
| `seed=3` | `cellular\|late` | 44,85% | `false` — explotou |
| `seed=53` | `telephone\|late` | 25,96% | **`true`** — explorou |
| `seed=65` | `telephone\|early` | 23,19% | **`true`** — explorou |

**Fale:** em 5% das chamadas o sistema **deliberadamente não** joga o melhor braço. Parece
desperdício, e é — a curto prazo. É o preço de continuar aprendendo, e é exatamente o que um teste
A/B congelado não faz.

**Diga também o que falta**, porque a banca vai perguntar: esse `explore` é sem estado. Aprendizado
online de verdade exige endpoint de feedback e as posteriores por braço persistidas — está desenhado
no Terraform em `infra/` (Firehose + DynamoDB), mas a aplicação ainda não usa.

---

## Passo 7 — Swagger, se sobrar tempo

Abra http://localhost:8000/docs, expanda `POST /recommend`, clique em **Try it out**. O exemplo já
vem preenchido. Vale mostrar que `duration` **não existe** no formulário — a coluna proibida pelo
enunciado não tem por onde entrar.

---

## A frase que fecha

> "O adaptativo supera o baseline em 18%. Mas medimos o teto do ganho contextual antes de assumir
> que personalizar valia a pena, e ele era de 4,4% — então implementamos, medimos o custo, e
> mostramos que nesses dados não se paga. Está tudo no README, com os números."

Isso é mais forte que só apresentar o resultado bom, e desarma a pergunta difícil antes que ela
venha.

---

## Se algo der errado

| Sintoma | Causa | Saída |
|---|---|---|
| `/health` diz `degraded` | Falta `models/environment.joblib` | `make train` |
| `curl` recusa conexão | API não subiu | Ver o terminal do `make api` |
| Números diferentes destes | O modelo foi retreinado | Normal — o `SEED` é fixo, mas mudança de versão de biblioteca move as casas decimais |
| MLflow vazio | Rodou com `--no-mlflow` | `make train` sem a flag |

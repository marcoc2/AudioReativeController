# Backlog: coleção de efeitos

Ideias de efeitos novos, para atacar um de cada vez. Cada um entra como
**layer source** (ou post-op) seguindo `docs/EFFECTS_GUIDELINES.md`.

A regra da coleção: o efeito é regido pela **partitura**, não só pelo volume.
Cada item diz o que cada papel musical (trilha MIDI, stem, acorde, forma)
faz com a imagem. Isso é o coração do efeito, e o resto é implementação.

Status: `[ ]` a fazer · `[~]` em andamento · `[x]` feito

---

## Definição de pronto (vale para todos)

- [ ] Roda na GPU (moderngl), com fallback só se for trivial; ≤ ~10 ms/frame em 720p
- [ ] Determinístico: mesma cena + seed ⇒ mesmo vídeo; movimento derivado de `t`, nunca de wall-clock
- [ ] Hits via `_layer_hits` (aceita `track:`, `notes:`, `audio:`); contínuos via `features_at` suavizados
- [ ] Registrado em `build_compositor()` com docstring de YAML no wrapper
- [ ] `tests/test_<efeito>.py`: shape/dtype, determinismo, um hit muda a imagem, gesto inválido levanta erro
- [ ] Cena de exemplo em `examples/` com o comando de render no cabeçalho
- [ ] Render de 8–9 compassos de uma música real revisado a olho (frames extraídos)

---

## Família orgânica (corpo)

A série começou com `eyes`. Tom: realista, úmido, perturbador.

### 0. [x] Base compartilhada "parede de carne" → `core/flesh.py`
Antes de bocas/colmeia, extrair do `core/eyes.py` o que é comum: Voronoi com
distância suave à borda, altura/normal da pele, poros, sulcos, SSS vermelho,
ACES. Vira um trecho GLSL incluído por vários efeitos.
- Deps: —
- Pronto quando: `eyes` usa a base e o render de `esfolado_olhos.yaml` não muda (comparar frames)

### 1. [x] `mouths`: parede de bocas → `core/mouths.py`, `core/voice.py`, `examples/esfolado_bocas.yaml`
Mesma parede, cada célula com lábios, dentes, língua e saliva.
| Papel musical | Gesto |
|---|---|
| stem de voz (volume) | abertura da boca |
| timbre da voz (formantes F1/F2) | formato: "a" aberto, "u" em bico, "i" esticado |
| caixa | dentes trincam (`clench`) |
| bumbo | lábios tremem |
| acorde | tom dos lábios / quão vermelho o interior |
- Tarefas: estimar F1/F2 do stem de voz (LPC por frame, cache como as outras features) · shader da boca (lábio com SSS, dentes com esmalte translúcido, cavidade escura, brilho de saliva) · wrapper `MouthsLayer`
- Deps: 0
- Música alvo: qualquer uma com voz forte; par natural com `eyes` numa mesma cena (alternar por `bars:`)

### 2. [ ] `flay`: esfolar
Superfície de pele que rasga a cada ataque, mostra músculo e fibras, e cicatriza.
| Papel musical | Gesto |
|---|---|
| bumbo | abre um rasgo (posição por hash, tamanho pela velocity) |
| caixa | rasgo mais raso e comprido (arranhão) |
| sustain dos pads / volume geral | velocidade de cicatrização |
| acorde | tom da carne exposta |
- Tarefas: campo de dano persistente em textura ping-pong (feridas acumulam e cicatrizam) · camadas pele → gordura → músculo fibroso, com borda de rasgo irregular · sangue escorrendo pela gravidade (opcional)
- Deps: 0
- Música alvo: **Esfolado**

### 3. [ ] `hive`: colmeia / casulos
Favo hexagonal de cera com larvas se mexendo sob o opérculo.
| Papel musical | Gesto |
|---|---|
| notas do baixo | larvas pulsam (cada nota, uma região) |
| ataques (qualquer trilha escolhida) | um opérculo rompe |
| volume | quanto se mexem |
| acorde | tom da cera (âmbar → escuro) |
- Deps: 0 (a grade é hexagonal, mas pele/SSS são reaproveitados)
- Música alvo: **Enxame**

---

## Família física (realismo)

### 4. [x] `ferrofluid`: ferrofluido → `core/ferrofluid.py`, `examples/esfolado_ferrofluido.yaml`
Líquido preto espelhado que forma espinhos sob um campo magnético.
| Papel musical | Gesto |
|---|---|
| subgrave / stem de baixo | altura dos espinhos |
| bumbo | pulso do ímã: espinhos disparam e voltam com mola |
| acorde | rotação/rearranjo do padrão de espinhos |
| cada trilha MIDI escolhida (bumbo, caixa, baixo…) | o seu ímã: um pulso de mola a cada nota |
| forma da música (`bars:`) | número de ímãs (1 no verso, vários no refrão) — *ainda não: hoje os ímãs são fixos* |
- Tarefas: campo de altura com espinhos em arranjo hexagonal deformado pelo campo · raymarch do heightfield · material: reflexo de ambiente cromado quase preto, fresnel forte · física simples (mola amortecida por espinho) em Python, uniforms na GPU
- Deps: —
- Por que primeiro: maior impacto visual, fica fotorrealista, lê o grave de forma óbvia

### 5. [ ] `ink`: tinta na água
Simulação de fluido 2D; cada trilha injeta tinta da sua cor.
| Papel musical | Gesto |
|---|---|
| cada trilha MIDI | injeção de tinta na sua cor e posição (o arranjo fica visível) |
| velocity | quantidade e força do jato |
| volume geral | turbulência (vorticidade) |
| acorde | paleta das trilhas |
- Tarefas: Navier-Stokes estável (advecção semi-lagrangiana + projeção de pressão, Jacobi) em texturas float · tinta com absorção (Beer-Lambert) sobre fundo claro, para parecer tinta e não luz · mapeamento trilha → cor/posição no YAML
- Deps: —

### 6. [ ] `mud`: lama rachando
Crosta que seca e racha em polígonos; lama borbulha embaixo.
| Papel musical | Gesto |
|---|---|
| bumbo | rachaduras se propagam (fratura por Voronoi hierárquico) |
| stem de baixo | bolhas de lama sobem e estouram |
| forma da música | umidade: no início lama mole, no fim deserto seco |
- Deps: —
- Música alvo: **Deserto de lama** (lembrar do `--meter 22:6/4,27:5/4`)

### 7. [ ] `murmuration`: revoada de estorninhos
Centenas de milhares de pássaros com comportamento de bando.
| Papel musical | Gesto |
|---|---|
| contorno da melodia (altura das notas do lead) | altura/forma da nuvem |
| caixa | o bando se parte e volta a se juntar |
| volume | densidade / velocidade |
| forma da música | ponto de atração muda por seção |
- Tarefas: boids na GPU (compute shader ou ping-pong de texturas, grade espacial para vizinhança) · render como pontos pequenos escuros com desfoque de profundidade sobre céu de fim de tarde
- Deps: — (não estender `core/particles*.py`, que é legado congelado)
- Música alvo: **Enxame**

---

## Família "a partitura vira objeto"

### 8. [ ] `strings`: cordas vibrando
Um instrumento imaginário com uma corda física para cada nota tocada.
| Papel musical | Gesto |
|---|---|
| cada nota MIDI (altura) | qual corda (posição/espessura) |
| velocity | amplitude do golpe |
| duração da nota | quanto tempo vibra antes de abafar |
| trilha | material da corda (aço, nylon, tripa) |
- Tarefas: onda 1D por corda com harmônicos e decaimento (analítico, sem simulação) · render 3D com reflexo metálico e desfoque de movimento na parte que vibra
- Deps: —

### 9. [ ] `slitscan` (post-op): atraso de tempo por linha
Cada linha da tela mostra um instante diferente do passado.
| Papel musical | Gesto |
|---|---|
| volume | quanto atraso (0 = imagem normal) |
| bumbo | onda de atraso atravessa a tela |
| forma da música | direção (vertical, horizontal, radial) |
- Tarefas: ring buffer de frames na GPU (textura 3D ou array) · `process(frame, t)` lê cada linha no frame certo
- Deps: — (funciona sobre qualquer camada, inclusive `eyes`)

---

## Ordem sugerida

```
4 ferrofluid                 (impacto rápido, independente)
0 base carne → 1 mouths      (série do corpo, par com eyes)
            → 2 flay         (Esfolado)
9 slitscan                   (post-op barato, multiplica tudo o que já existe)
5 ink, 6 mud, 7 murmuration  (conforme a próxima música pedir)
3 hive, 8 strings
```

A ordem cede à próxima música: se o clipe da vez for **Deserto de lama**,
`mud` passa na frente.

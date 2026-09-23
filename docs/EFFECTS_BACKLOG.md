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

### 9. [x] `slitscan` (post-op): atraso de tempo por linha → `core/slitscan.py`, `examples/esfolado_olhos_slitscan.yaml`
Cada linha da tela mostra um instante diferente do passado.
| Papel musical | Gesto |
|---|---|
| volume | quanto atraso (0 = imagem normal) |
| bumbo | onda de atraso atravessa a tela |
| forma da música | direção (vertical, horizontal, radial) — *hoje: uma camada `slitscan` por direção, cada uma com seu `bars:`* |
- Tarefas: ring buffer de frames na GPU (textura 3D ou array) · `process(frame, t)` lê cada linha no frame certo
- Deps: — (funciona sobre qualquer camada, inclusive `eyes`)

---

## Família "o vídeo como matéria" (lê os frames dos `--clips`)

Efeitos que analisam o quadro composto até ali (os clipes, ou qualquer camada
abaixo) e extraem dele algo para brincar: os **contornos** ou um **mapa de alturas**
(brilho do pixel = altura). Entram como post-op (`process`) ou como camada que se
pinta com o que está embaixo (`frame_at_over`, como a chama com `paint: under`).
Pensados primeiro para clipes cartoon, cujo traço limpo dá contornos fechados.

### 10. [~] Base compartilhada "leitura do quadro" → `core/frame_read.py` (contornos: feito; regiões e altura: a fazer)
Um passe de GPU que recebe o frame e devolve, em texturas: contornos (Sobel ou
diferença de gaussianas, com limiar e engrossamento opcional), máscara de regiões
fechadas, e altura suavizada (luminância com blur). Os itens abaixo leem daqui.
- Tarefas: `core/frame_read.py` sobre o `ShaderPass` (upload do frame, passes em ping-pong) · parâmetros de limiar/espessura no YAML · testes com imagens sintéticas (quadrado desenhado, gradiente)
- Deps: —
- Pronto quando: um desenho sintético dá contornos fechados, e um gradiente dá uma rampa de altura

### 11. [x] `drops`: gota que se espalha até os contornos → `core/drops.py`, `examples/esfolado_gotas_curadoria.yaml`
A cada ataque cai uma gota num ponto; ela pega a cor do pixel onde caiu e se
espalha, como um balde de tinta lento, até bater nos contornos do desenho.
| Papel musical | Gesto |
|---|---|
| caixa (ou trilha escolhida) | solta uma gota (posição por hash, ou no ponto mais claro/escuro) |
| velocity | velocidade de espalhamento |
| duração da nota / envelope | quanto tempo a mancha fica antes de secar e sumir |
| acorde | cor da gota: a do pixel onde caiu, ou a da paleta do acorde |
- Tarefas: crescimento de região na GPU (a mancha dilata alguns pixels por frame e para na máscara de contornos) · várias manchas vivas ao mesmo tempo, cada uma com cor e idade · borda da mancha com leve brilho/umidade · o contorno pode deixar vazar um pouco (limiar), para gotas atravessarem traços finos
- Variação `shape: spiral` (`examples/esfolado_gotas_espiral_curadoria.yaml`): a tinta enche a região do mesmo jeito, mas só aparece num braço de espiral que se enrola para fora do ponto da gota (`pitch`, `turns`, `arm`, `spin`)
- Deps: 10

### 12. [ ] `neon_lines`: só os contornos, em neon
| Papel musical | Gesto |
|---|---|
| acorde | cor do neon |
| compasso (fase) | o traço vai sendo desenhado ao longo do compasso |
| bumbo | pulso de brilho (glow) |
- Deps: 10

### 13. [ ] `recolor`: recolorir o cartoon por regiões
Cada região fechada pelos contornos ganha uma cor chapada; a paleta troca a cada acorde.
| Papel musical | Gesto |
|---|---|
| acorde | paleta (a região mantém o "índice" de cor, a paleta muda) |
| caixa | embaralha quais regiões ficam com qual cor |
- Tarefas: rotular regiões (componentes conexos na GPU por propagação de rótulo, ou na CPU em baixa resolução) · manter o rótulo estável entre frames (casar regiões pelo centroide)
- Deps: 10

### 14. [x] `disintegrate`: o desenho vira poeira → `core/sandlines.py` (`source: sandlines`), `examples/esfolado_areia_curadoria.yaml`
Partículas nascem sobre os contornos, com a cor do traço; o bumbo as sopra para longe e o desenho se refaz.
- Feito como areia branca: o primeiro frame do clipe de cada compasso é invertido, o contraste esticado, e o quase branco (o traço) vira areia; a imagem some. A cada compasso a areia escorre para as linhas do desenho novo. A onda de choque do item 20 colore a areia e a arremessa para fora e para cima (física: voo com gravidade, quique, atrito de Coulomb no chão); a vibração da placa traz de volta ao traço (`pull`). Um kick faz a placa inteira pular (`jump`). Simulação na GPU, 200 mil grãos, ~3 ms/frame
- A fazer: grãos com a cor do traço original (hoje brancos) como opção
- Deps: 10 (as partículas na GPU; não estender `core/particles*.py`, legado congelado)

### 15. [ ] `relief`: o vídeo esculpido
O frame vira superfície 3D (barro, metal batido); a luz gira no ritmo; a câmera inclina com o grave.
- Deps: 10 (altura)

### 16. [ ] `pinscreen`: brinquedo de pinos
Uma grade de pinos empurrados pelo brilho do vídeo; o bumbo empurra todos de uma vez.
- Deps: 10 (altura)

### 17. [ ] `ferrofluid` pintado pelo vídeo
Extensão do item 4: o campo magnético vem do brilho do vídeo (os espinhos crescem onde é claro) e o reflexo usa as cores do quadro.
- Deps: 4, 10 · o que menos dá trabalho: o motor já existe

### 18. [ ] `ripples`: superfície de água sobre o vídeo
Cada batida cai uma gota; as ondulações (equação de onda de verdade, ping-pong na GPU) entortam o vídeo por refração e dão brilho especular.
- Deps: — (combina com 11: a mesma gota pode espalhar cor e ondular)

### 19. [ ] `topo`: mapa topográfico
Curvas de nível do brilho do vídeo; as linhas sobem e descem com o grave.
- Deps: 10 (altura)

### 20. [x] `shockwave` (post-op): onda de choque nos contornos → `core/shockwave.py`, `examples/esfolado_onda_curadoria.yaml`
Vem do gesto `ring` da areia (`core/chladni.py`, `_gesture`), que ficou ótimo no trecho
de piano do `esfolado.mp4` (compassos 22–35): a cada nota um anel nasce e se expande
(~0,6 s), dobra a imagem por onde passa como uma lente e tinge **só as linhas** da
figura (a areia branca vira rosa), deixando o fundo escuro como está. Aqui o mesmo,
sobre qualquer imagem, pensado para cartoons: as cores do cartoon são **invertidas**,
então o traço preto vira linha branca sobre fundo escuro, como a areia, e a onda
atravessa o desenho.
| Papel musical | Gesto |
|---|---|
| trilha escolhida (no esfolado era o piano, `track: 7`) | cada nota solta um anel; acordes soltam anéis sobrepostos |
| velocity | força do empurrão e do brilho do anel |
| acorde | cor que a onda dá às linhas |
| bumbo (opcional, como o `punch` do `kick-2`) | a imagem toda incha um pouco e dá um flash |
- Tarefas: inversão opcional (`invert: true`) antes de tudo · anel gaussiano como na areia (velocidade, largura, vida, empurrão radial) virando deslocamento de pixels (lente) na GPU · tingir só onde há linha: máscara pelo brilho (linhas claras depois da inversão) ou pelos contornos do item 10 · vários anéis vivos ao mesmo tempo · centro do anel: meio da tela (como na areia), ponto por hash, ou onde caiu a gota do item 11
- Referência dos números da areia: `RING_SPEED = 3.2`, `RING_WIDTH = 0.12`, `RING_PUSH = 0.06`, `RING_LIFE = 0.6` (unidades de campo: a tela tem 2 de altura)
- Deps: — (máscara por brilho); 10 melhora a máscara em vídeos que não são cartoon

---

## Ordem sugerida

```
4 ferrofluid                 (impacto rápido, independente)
0 base carne → 1 mouths      (série do corpo, par com eyes)
            → 2 flay         (Esfolado)
9 slitscan                   (post-op barato, multiplica tudo o que já existe)
5 ink, 6 mud, 7 murmuration  (conforme a próxima música pedir)
3 hive, 8 strings

10 leitura do quadro → 11 drops   (vídeo como matéria: a gota primeiro)
                    → 12 neon_lines, 13 recolor, 14 disintegrate
                    → 15 relief, 16 pinscreen, 19 topo
4 + 10 → 17 ferrofluid pintado  (resultado rápido: o motor já existe)
18 ripples                      (independente; par natural da 11)
20 shockwave                    (independente; cartoon invertido + a onda da areia)
```

A ordem cede à próxima música: se o clipe da vez for **Deserto de lama**,
`mud` passa na frente.

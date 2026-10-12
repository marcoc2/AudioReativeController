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

## Família "as figuras do vídeo" (máscaras SAM2)

O render base vira uma **trilha de máscaras**: o SAM2 separa cada plano em objetos
(as figuras) e grava isso uma vez só, quadro a quadro. Os efeitos leem as máscaras pelo
tempo da música. As figuras viram a banda: cada instrumento ganha um corpo na tela.

Pipeline (o comando completo fica no cabeçalho de cada cena):

```
clip_generator.py (só "- source: clips")  →  render base + <stem>_cuts.json (os cortes)
segment_video.py --video <base>           →  <base>_segments/ (máscaras; --preview pra ver)
clip_generator.py --scene <cena> --masks <base>_segments
```

As máscaras são PNGs de 16 bits com o id de cada objeto. Num visualizador de imagem
parecem pretas; para vê-las, use o `_segments_preview.mp4`. Os cortes vêm do
`_cuts.json` que o render grava. Detectar cortes pela imagem confunde clipes que
piscam com cortes.

### 21. [x] Trilha de máscaras → `core/segment/mask_track.py`, `segment_video.py`
SAM2 por plano: objetos acompanhados pelo plano inteiro. Cada objeto tem id, tamanho
médio e posição.

### 22. [x] `mask:` em qualquer camada → `core/video/matte.py`, `examples/esfolado_world_melts.yaml`
`where: inside | outside`, `objects`, `grow`, `feather`. O mundo derrete (feedback,
shader, slitscan `outside`) e as figuras ficam nítidas, ou o contrário.

### 23. [x] `cast`: elenco por papel → `core/video/cast.py` (GPU + numpy)
Os papéis são preenchidos de novo em cada plano. `who` aceita:
- tamanho: `largest`, `second`, `third`, `fourth`;
- posição: `leftmost`, `rightmost`, `topmost`, `bottommost`;
- `all`, `background` ou uma lista de ids.

O papel toca com `plays`, que é um trigger comum (`track`, `notes`, `audio`, `score`).

| Ato | Gesto | Cena |
|---|---|---|
| `punch` | salta do papel na batida, maior, com sombra | `esfolado_cast` |
| `shake` | tranco lateral, deixa o fantasma | `esfolado_cast` |
| `tint` | ganha cor mantendo o sombreado (acorde ou cor fixa) | `esfolado_cast` |
| `glow` | halo no contorno | `esfolado_cast` |
| `presence` | silhueta chapada até o instrumento tocar | `esfolado_cast` |
| `rings` | anéis saem do contorno como sonar (grave grosso, forte brilhante) | `esfolado_melodia` |
| `echo` | fantasmas enquanto a nota soa, rastro proporcional à duração | `esfolado_melodia` |
| `carry` | atravessa o corte como adesivo até a próxima batida | `esfolado_colagem` |
| `wake` | rastro temporal: onde a figura passou fica e se apaga | `esfolado_rastro*` |

Detalhes do `wake`:
- **`hold`**: quanto tempo o rastro fica ligado. `note` = enquanto a nota soa; `bar` = até o fim do compasso; um número = segundos.
- **`length`**: segundos até o rastro cair a ~5%.
- **`every`**: amostragem, em segundos ou como valor de nota na grade (`1/16`). Vira um estroboscópio de cópias; sem ele, o rastro carimba todo quadro.
- **`shift` + `direction`**: o rastro deriva em pixels/s, e a cada clipe a direção gira pelo ângulo áureo.
- **`amount`**: opacidade. **`color`**: tinge o rastro.
- Um corte apaga o rastro.

Todas as cenas juntas: `esfolado_completo`, `carrossel_completo`, `carrossel_retrato`,
`carrossel_abstrato` e `me_diz_capa`.

### 24. [ ] Próximos atos e ideias
- [ ] `stutter`: a figura engasga (repete alguns quadros) na batida. É o mais barato.
- [ ] O acorde escolhe quem acende (cada figura com uma nota do acorde).
- [ ] Wipe pela silhueta: a troca de clipe acontece através do contorno da figura.
- [ ] Parallax 2.5D: figuras e fundo em planos que se deslocam na batida.
- [ ] Partículas que saem do contorno.
- [ ] Repintura mascarada no ComfyUI (só o fundo, ou só as figuras; ver `esfolado_repaint`).
- [ ] Filtro de balões de fala, que hoje às vezes viram personagens.
- [ ] Escolher figuras por classe (`who: person`): caixas do YOLO como prompts do SAM2. Fica para quando houver vídeo com pessoas.
- [ ] `wake` atravessando o corte (como o `carry`), para rastros mais longos que o plano.

---

## Da pesquisa de out/2026: outros visualizadores e fractais

Três levantamentos foram feitos:
- visualizadores audio-reactive: MilkDrop/Butterchurn, Hydra, Synesthesia, parseq e TouchDesigner;
- fractais 2D: flames, Julia/Newton/Lyapunov, Droste e Kleinianos;
- fractais 3D: Mandelbulber, Fragmentarium, KIFS e Marble Marcher.

As ideias abaixo são as que **não** estavam no backlog. Elas estão agrupadas pelo papel
da música, e não pela técnica.

**Licenças.**
- Mandelbulber, Fragmentarium, Marble Marcher, flam3/Fractorium e XaoS são GPL.
- A Hydra é AGPL.
- Os shaders do Shadertoy são CC BY-NC-SA por padrão.

Em todos esses casos, escrever do zero a partir da matemática. Butterchurn (MIT) e
jsexp (BSD) podem ser aproveitados com atribuição.

**Caminho rápido.** A maioria cabe num `.glsl` nosso (`_arc`) no adaptador shadertoy,
ligado à música por `uniforms:`, sem Python novo. Depois vira camada própria, se valer.

### 25. A base: como a música vira número
- [x] Knob `chord:` → `core/video/layers.py`. É o acorde que soa agora: `root`, `tone` (terça, quinta…), `size`, `mode` (maior/menor/sus) e `tension`, com `glide`.
- [x] Knob `cut:` → `layers.py` + `ClipComposer.cut_times`. É o corte de clipe, conhecido **antes** de chegar: `count`, `since`, `until` e `phase`.
- [ ] `channel: mask` no shadertoy: a máscara do SAM2 (ou o campo de distância dela) como textura do shader.
- [ ] **Intensidade relativa.** Hoje cada banda é medida contra o máximo da música, o que deixa a estrofe parada e satura o refrão. Medir contra a média recente, com ataque rápido e soltura lenta, como faz o `audioLevels.js` do Butterchurn. Somar "presença" (atividade sustentada) e "intensidade" (a construção lenta da música), como no Synesthesia.
- [ ] **Envelope ADSR com note-off.** Hoje o `EnvelopeOpacity` só decai a partir do ataque. O envelope deveria segurar enquanto a nota soa e soltar no fim dela, com a velocity escalando o pico, como o Trigger CHOP do TouchDesigner.
- [ ] **Antecipação.** O render é offline: um efeito pode começar antes da batida, como o pré-corte ou a tensão no último tempo. O `cut: until` já abre esse caminho para os cortes.
- [ ] **Moduladores em qualquer parâmetro**, não só nos `uniforms:` do shadertoy. Exemplos: `max_rotate: {tri: 2b}` e `hue_shift: {random_on: {track: snare}, ease: 0.1}`. Modelos: os osciladores em batidas do parseq e o `syn_RandomOnBeat` do Synesthesia.

### 26. O corte como gesto (transições)
Hoje o corte é seco. O knob `cut:` dá ao efeito o tempo até o corte. Para misturar dois
clipes, o efeito precisa segurar o último quadro do plano que sai, como o `carry` faz.
- [ ] **Padrões do MilkDrop:** o clipe novo entra por uma frente que varre num ângulo, sai do centro ou vem em manchas de plasma. Fórmula: `clamp(p*(1+b)/b - (1-t)/b, 0, 1)` (Butterchurn `blendPattern.js`, MIT). O avanço segue o último tempo antes do corte, e a bateria escolhe o padrão. Variante ARC: o campo é a distância às figuras, então elas trocam primeiro e o mundo depois. É mais geral que o wipe pela silhueta do item 24.
- [ ] **Portal Menger:** o último quadro vira uma face de Menger, e a cada batida abre mais um nível de buracos por onde o clipe novo aparece.
- [ ] **Wipe pelas curvas de nível de um fractal** (Julia ou Lyapunov), com o `c` vindo do acorde do corte.
- [ ] **Droste como transição:** o clipe novo mora na cópia interna da espiral, e o corte cai quando a câmera chega nela.
- [ ] **Datamosh:** os pixels do plano velho são arrastados pelo movimento do novo, até a caixa "limpar" a imagem. O fluxo ótico é calculado na CPU (OpenCV DIS) e guardado por clipe; o tamanho do bloco vem da velocity. Dá muito trabalho e também muito resultado.

### 27. A harmonia como cor e forma
- [ ] **Paleta pelo acorde** (pós-efeito): a luminância do quadro passa por uma rampa feita das notas do acorde. A qualidade define luz e saturação: maior mais claro, saturado e amarelo; menor mais escuro e azul (Palmer et al., PNAS 2013). A rampa troca em um tempo. Usar a cor do baixo no fundo e a do acorde nas figuras, via `mask:`.
- [ ] **Newton/Nova cujas raízes são as notas do acorde**, nos ângulos da roda de 12. Numa troca de acorde as raízes deslizam e as bordas se retrançam. Cada bacia ganha a cor da sua nota. Variante: cada bacia mostra o clipe girado (um caleidoscópio do acorde).
- [ ] **KIFS 3D com simetria pelo acorde:** icosaédrica para maior, octaédrica para menor, tetraédrica para dim/aug. A tônica gira as dobras e o kick infla a escala.
- [ ] **Orbit traps harmônicos** (um include GLSL para todo fractal, inclusive `mandelbulb` e `mandelbox`): tônica, terça e quinta em três canais de trap, e uma nota da melodia acende uma quarta.
- [ ] **Notas como luzes dentro do fractal:** cada nota é um ponto de trap brilhante, com velocity e duração (névoa por iteração e "fake lights" do Mandelbulber). A partitura vira luz dentro do objeto.
- [ ] **Julia quaterniônico:** o acorde move `c` e a melodia percorre a fatia 4D.

### 28. A bateria como estrutura
- [ ] **Lyapunov:** a sequência A/B é a bateria do compasso em semicolcheias (bumbo = A, caixa = B). Cada levada tem a sua paisagem, e uma virada a muda. Num compasso de outro tamanho a sequência muda (Deserto: 5/4 → 6/4).
- [ ] **Fractal 3D híbrido:** os 16 passos do compasso escolhem a fórmula de cada iteração (bumbo = dobra Mandelbox, caixa = bulb, hi-hat = dobra KIFS). Cada levada vira uma "espécie".

### 29. O fractal deformando o clipe
- [ ] **Variações do flame como lente sobre o vídeo:** swirl, bubble, spherical e outras, cada peso vindo de um stem. Reaproveita `core/flame.py`.
- [ ] **Droste:** o quadro contém a si mesmo numa espiral, com uma volta por compasso, o que dá um loop perfeito. A cópia interna fica na caixa da figura.
- [ ] **Julia com orbit trap feita da figura** (o campo de distância da máscara) ou do próprio quadro: o personagem se repete em todas as escalas.
- [ ] **MRCM, a "copiadora":** um IFS cujo atrator é feito de cópias do clipe. Cada trilha MIDI é uma lente, e a batida reforça a cópia dela.
- [ ] **Ladrilhamento hiperbólico {p,q}** (Escher, *Circle Limit*): `p` vem do numerador do compasso e `q` do tamanho do acorde, com um ladrilho de translação por tempo.
- [ ] **Kleinianos / Indra's Pearls:** o clipe dentro de cada pérola, com a melodia movendo `µ`.
- [ ] **Domain warp** (fbm∘fbm∘fbm do iq) como deslocamento do vídeo: baixo no primeiro estágio, voz no segundo.
- [ ] **Modulação cruzada** (`modulate`/`modulateHue` da Hydra): uma imagem entorta a outra. Por exemplo, o clipe que entra entorta o que sai no último tempo.
- [ ] **Campo de warp do MilkDrop:** zoom por raio, rotação por ângulo e turbulência, com o centro no centróide da figura principal (o vórtice segue quem canta). Hoje o `feedback` é só um zoom global.
- [ ] **Fractal 3D com o clipe como textura:** projeção triplanar, ou o UV vindo da última iteração. Com a máscara, as figuras viram estátuas repetidas na superfície.

### 30. Mundos de fundo (atrás das figuras, com `mask: outside`)
- [ ] **Star Nest** (Kali): nuvens volumétricas. É quase só o `.glsl`, com licença MIT a confirmar.
- [ ] **Fatiador:** um plano corta o fractal uma vez por compasso, como uma tomografia.
- [ ] **"Fases" do Marble Marcher:** o fractal muda de forma a cada corte (`cut: count`), e as animações senoidais andam presas ao compasso.
- [ ] **Voo por waypoints:** pseudo-Kleiniano ou Amazing Surf, um waypoint por compasso e um galho novo a cada corte.
- [ ] **Bolhas apolonianas:** o kick infla a inversão e as notas acendem as cascas. Combina com o `soap_bubble.glsl`.
- [ ] **Nebulabrot pelos stems** (baixo, bateria e voz em três tetos de iteração): reaproveita o caminho de histograma do flame.
- [ ] **Zoom profundo ao longo da música inteira** (perturbação): uma década a cada N compassos, chegando a um minibrot no refrão.
- [ ] **Gray-Scott** semeado pelo quadro: o acorde salta (f, k) entre manchas, vermes e labirintos.
- [ ] **Melhorias no flame:** um xform por trilha MIDI e a estimativa de densidade do flam3.
- [ ] **Desfoque de movimento e profundidade de campo por sub-quadros** em qualquer fractal da GPU.

### 31. Pós-efeitos musicais
- [ ] **Mapa de deslocamento no tempo** (Time Machine TOP do TouchDesigner): cada pixel lê um quadro passado escolhido por um mapa (distância à figura, luminância ou ruído), com o atraso em valores de nota. Inclui o split RGB no tempo: R agora, G uma colcheia atrás, B uma semínima atrás.
- [ ] **Bloom musical** (blur1/2/3 do MilkDrop): a batida baixa o limiar e sobe o ganho, e o acorde tinge o brilho.
- [ ] **Pixel sorting:** o rastro cresce enquanto a nota soa e derrete no note-off, com o limiar seguindo a tensão do acorde.

Novos atos do elenco, pelo item 24:
- [ ] `clock`: a figura só anda enquanto o instrumento dela toca e congela nas pausas (como o `syn_BassTime` do Synesthesia, mas com o MIDI exato).
- [ ] `sing`: a figura sobe e desce com a melodia, deslizando no legato.
- [ ] `scope`: o contorno vira um osciloscópio, com ondas sintetizadas das notas MIDI. Um acorde dá interferência no estilo Lissajous.
- [ ] Reação-difusão crescendo dentro da figura: cada nota semeia um ponto dentro da máscara.

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

21 máscaras → 22 mask: → 23 cast (feitos) → 24 stutter primeiro

25 knobs chord + cut (feitos) → 27 paleta pelo acorde (pós, muda tudo de uma vez)
                             → 26 transições no corte (padrões MilkDrop + variante pelas figuras)
                             → 29 lente de variações do flame (reaproveita o flame)
                             → 27 KIFS pelo acorde + orbit traps harmônicos
25 intensidade relativa + ADSR  (melhora todos os efeitos que já existem)
```

A ordem cede à próxima música: se o clipe da vez for **Deserto de lama**,
`mud` passa na frente.

# Backlog da GUI (ARC Studio)

O objetivo: que alguém que não lê YAML consiga conceber pela GUI qualquer cena que hoje
está em `examples/`, escolhendo o que na música move o quê e vendo isso antes do render.

Saiu de um exercício (2026-10-13): percorrer na GUI a criação de três cenas que cobrem os
tipos que existem: um shader com knobs (`me_diz_perolas`), o elenco (`me_diz_canto`) e os
triggers de montagem (`deserto_kick_reverse`). O vocabulário dos 108 YAMLs de exemplo:
19 formas de knob (~470 usos: hits 131, chord 48, harmony 47, feature 64, relative 18,
lfo 16, voice 15, cut 14, pitch 11, groove 10, adsr 10, …), 11 atos do elenco, ~100
parâmetros de camadas nativas, 125 âncoras YAML (o mesmo sinal usado em dois lugares).

Onde a GUI está hoje: insumos (áudio, MIDI, offset, compasso), render e triggers de corte
funcionam; a pilha de camadas tem estrutura (fonte, blend, opacidade, compassos), mas
todo o conteúdo que faz a cena (knobs, elenco, máscaras, painel `meters`) é YAML
digitado num campo de texto, e cenas com elenco/máscara não renderizam pela GUI.

Ordem: primeiro o que estraga ou bloqueia (1–4), depois o centro da abstração visual
(5–9), depois o resto.

## 1. [x] "Save Scene" estraga os YAMLs

**Como aparece.** Reload Scene + Save Scene numa cena de exemplo sem mudar nada:
os comentários inline somem, as âncoras viram `&id001`, e cada camada vira um bloco
separado, então o `*kick` do painel `meters` vira uma cópia: mudar o knob do shader deixa
de mudar a lane.

**Correção.** A cena é carregada com `ruamel.yaml` (ida e volta preserva comentários,
âncoras e estilo) e o save aplica só a diferença entre o que foi carregado e o que está
nos widgets, no lugar. O texto de parâmetros de cada camada mostra as âncoras de outras
camadas como `*nome`, e elas continuam ligadas.

## 2. [x] Render e preview ignoram as máscaras

**Como aparece.** `_build_render_cmd` nunca passa `--masks`; o preview não passa
`masks` ao `build_compositor`. As 13 cenas de elenco e as que usam `mask:` falham.
Também faltam `--clip-turn-wide` e `--profile`.

**Correção.** Campo "Masks…" (pasta `*_segments`) salvo no projeto, passado ao render e
ao preview; checkboxes para turn-wide e profile.

## 3. [x] Preview no aspecto errado

**Como aparece.** O preview é sempre 596x336 (16:9): uma cena 480x854 aparece deitada.

**Correção.** Renderizar o preview no aspecto da resolução escolhida, encaixado no
painel com barras.

## 4. [x] Montagem sem widgets

**Como aparece.** `bars_per_clip` (47 cenas), `clip_end` (46), o `seed` da cena e o
`exclude:` dos triggers só sobrevivem se já estavam no arquivo. O início é em segundos
(o 18.3 do DAW precisa do scrubber). Não há presets em retrato nem "New Scene".

**Correção.** Widgets para esses campos; início também em compasso.tempo; presets
480x854, 720x1280, 1080x1920; botão New Scene (arquivo novo com `- source: clips`).

## 5. [x] Uniforms do shader à vista

**Como aparece.** Para saber que `perolas_arc.glsl` tem `u_s`, `u_kick`, `u_depth`,
`u_zoom` e o que fazem, é preciso abrir o .glsl.

**Correção.** Ler os `uniform float u_x = default;` (e as notas do shader) e mostrar
uma linha por uniform, com o default; a linha é o lugar do editor de knob (item 6).

## 6. [x] Editor de knob

**Como aparece.** As 19 formas de knob são digitadas à mão, e só se vê a curva depois
do render (painel `meters`).

**Correção.** Cada uniform/parâmetro ganha: "o que da música" (as formas agrupadas por
significado: batida, nota, harmonia, corte, textura do áudio, tempo/compasso, figura),
"de onde" (track/notas/stem/score, o mesmo widget dos triggers), "como responde"
(envelope, glide, gain/bias, low/high), aninhamento para `rate.from`, e a curva
desenhada na timeline.

## 7. [x] Sinais nomeados

**Correção.** Um painel de sinais do projeto ("kick", "organ", "acorde") definidos uma
vez e escolhidos em qualquer knob, papel do elenco ou lane: são as âncoras YAML
(`&kick`), editáveis num lugar só.

## 8. [ ] Timeline musical

**Correção.** Lanes de acordes (SheetSage, com botão para rodar o SheetSage), de
cortes (o cronograma do ClipComposer) e das curvas dos knobs escolhidos.

## 9. [ ] Painel `meters` automático

**Correção.** Um checkbox "mostrar os sinais no vídeo" que acrescenta a camada `meters`
com uma lane por knob da cena (a mesma âncora), em vez de escrever as lanes.

## 10. [ ] Elenco e máscaras

**Correção.** Botão "Segmentar" (render base + `segment_video.py`, com o check de GPU);
overlay no preview numerando as figuras (largest/second/third) de cada plano; tabela de
papéis (quem, toca, ato, e os parâmetros de cada ato); widget de `mask:` (onde, grow,
feather).

## 11. [ ] Formulários por fonte

**Correção.** Os parâmetros das camadas nativas (cells, flame, julia, feedback, …) viram
widgets a partir de um schema por fonte; o YAML livre fica como escape.

## 12. [ ] Solo de camada no preview

**Correção.** Um botão "solo" por camada: o preview mostra só ela sobre os clipes (como
os demos isolados), sem mexer na cena.

## 13. [x] Arranjo: escolher os clipes e quando entram fundo e personagens

**Problema.** O fundo era sorteado de uma pasta (no máximo um clipe fixado por compasso,
sem escolher o trecho), e os personagens só entravam escrevendo `bars:` e `cues:` no
YAML. Não dava para dizer "do compasso 9 ao 12, este clipe a partir dos 42 s" nem
"o leão entra no 5 e sai no 17" pela GUI.

**Correção.** Uma janela "Arranjo" (botão no editor de cena):
- navegador de clipes: pasta, filtro (ex.: `final.mp4`), miniatura, ponto de entrada
  (`from`) escolhido vendo o quadro;
- faixas sobre a régua de compassos: uma por camada `footage`. Fundo: um clipe por
  bloco; personagem: o seu clipe, recortado pela sua mask track. Arrastar cria um bloco,
  clicar seleciona, embaixo edita (compassos, from, velocidade) ou apaga;
- os cortes vindos do REAPER (`cues: {reaper: ...}`) aparecem em cinza, só leitura;
- "+ fundo", "+ personagem" (do clipe escolhido, achando a mask track em
  `render_output/footage/`), "Segmentar (SAM2)" quando não houver;
- salvar grava `blocks:` na cena pelo SceneDoc (comentários e âncoras preservados).

Formato: `blocks: [{bars: [9, 12], file: ..., from: 42.0}]` numa camada `footage`
(compassos do DAW, inclusivos; `file` por bloco no fundo).

**Feito (16/10/2026).** `arc_arrange.py` (a janela, botão "Arrange..." no editor de
camadas), `core/arrangement.py` (o modelo, testado), `blocks:` na camada `footage`.
Faixas que entravam por `bars:` aparecem como blocos e passam a `blocks:` ao serem
arranjadas. O personagem novo recorta tudo menos o fundo que o SAM2 tomou por objeto
(objeto com mais de 45% do quadro). Os cortes do REAPER ficam em cache
(`render_output/.cache/reaper_cues`). O SceneDoc agora leva os comentários junto com a
camada quando camadas entram ou saem. Arrastar um bloco o move, puxar uma ponta o
encurta ou alonga (a ponta da frente avança no clipe junto). Ctrl+Z / Ctrl+Y (e botões
Undo/Redo) desfazem e refazem cada edição, inclusive depois de salvar; só o Reload apaga o
histórico. Uma faixa de fundo sem blocos é salva como `blocks: []` e não mostra nada. Falta: zoom na régua,
posição/escala que seguem a música editadas na janela (hoje só números fixos; knobs pelo
editor de camadas), pré-visualizar só o trecho de um bloco.

## 14. [ ] Clips, Generators e FX na timeline (o arranjo como um REAPER)

**Ideia (16/10/2026).** O Arranjo (item 13) controla por compassos o que fica no fundo e
o que fica na frente, mas só com clipes. Muitos efeitos criam imagem (um fractal, células,
uma nebulosa); outros só fazem sentido sobre uma imagem. A GUI deve listar todos os
efeitos do projeto, agrupados por categoria, e deixar escolher um pelo botão direito em
cima da faixa (como inserir um plugin VST numa track do REAPER); o plugin abre uma janela
onde cada parâmetro é ligado a um elemento da música (kick, caixa, amplitude, notas,
acordes...).

**Os nomes** (na GUI em inglês):

| nome | o que é | no REAPER |
|---|---|---|
| Clip | um mp4 ou imagem | item de áudio |
| Generator | shader ou processamento que cria a imagem (fractal, células, nebulosa) | instrumento VST |
| FX | processa a imagem que recebe | efeito VST |
| Track | uma faixa da timeline (fundo ou personagem) | track |
| Master | os FX sobre o vídeo inteiro | master |

Os que fazem as duas coisas (o `flame` com `paint: under`, os shaders kifs/newton do
acorde com `u_video`) são Generators que aceitam uma imagem de entrada, como um sidechain.

No motor isso já existe sem nome: uma camada com `process()` é um FX (recebe a composição
até ali); com `frame_at()` é um Generator; com `frame_at_over()` é um Generator que vê o
que está embaixo. Um shader é FX quando alguma aba lê `frame`.

**Fases.**
1. **Catálogo** (`core/catalog.py`): todos os Generators e FX (as fontes nativas e os
   shaders), com tipo, categoria por significado, descrição e parâmetros (os `uniform` dos
   shaders; o exemplo YAML da docstring das camadas nativas); um teste garante que nada
   fica de fora. No Arranjo, um navegador em árvore com busca.
   **Feito (16/10/2026):** 70 entradas (34 camadas nativas, 36 shaders; um shader baixado
   com cópia `_arc` não aparece: a cópia fala por ele). Abas Generators e FX no painel
   esquerdo do Arranjo: categorias, busca, descrição ao passar o mouse, parâmetros com
   `~` nos que a música pode mover. `tests/test_catalog.py` falha quando uma fonte ou
   shader novo fica sem categoria.
2. **Generators na timeline + botão direito**: numa faixa vazia "Insert generator ▸
   categoria ▸ ..." ou "Insert clip"; numa faixa "Add FX ▸ ..."; uma faixa Master para os
   FX sobre tudo. Cada Generator/FX tem blocos de compassos (`bars:`), como os clipes.
   O motor já suporta: é só GUI.
   **Feito (16/10/2026):** toda camada da cena é uma faixa, o topo da imagem em cima
   (GEN, FX, BG/FG dos clipes, CLIPS da montagem embaixo de tudo). Botão direito numa
   faixa: "Insert generator above", "Add FX above", ligar/desligar (bypass, `enabled:
   false`), apagar; em qualquer lugar: "Insert generator as the background", "Add FX to
   the master", "+ background clip track". Os FX que só têm FX acima deles aparecem como
   "master: over everything"; os outros, "on the tracks under it" (o FX trabalha em tudo
   o que está embaixo dele até a fase 4). Arrastar numa faixa de Generator/FX dá
   `bars: [[22, 35]]` (sem blocos: a música inteira). O inspector de um Generator tem
   blend e opacidade; a timeline rola e o inspector fica à vista. As abas do navegador
   têm "Insert as background" / "Add to master".
3. **Janela do plugin**: cada parâmetro com valor e botão **Link** (ligar à música): os
   grupos de `knob_forms` (batida, nota, harmonia, corte, áudio, tempo, figura), os sinais
   nomeados do projeto (kick, snare) à mão, e a curva sobre os compassos. Reaproveita o
   editor de knob (item 6) e o painel de sinais (item 7). Um **Mix** (seco/molhado) por FX.
   **Feito (16/10/2026):** "Open plugin" (botão direito, o inspector, ou clique duplo no
   nome da faixa) abre a janela do Generator/FX: ligar/desligar, blend e opacidade (Generator),
   **mix** (FX), e cada parâmetro com o valor (o default em cinza), **Link** (o editor de
   knob do studio, com a curva sobre a música) e **x** (volta ao default). Um knob pode ser
   nomeado como sinal (`&kick`) e usado em outro plugin (`*kick`); mudar o sinal onde ele é
   definido (o primeiro na ordem da pilha) muda todos que o usam. No motor: `mix:` num FX
   (número ou knob, 0..1). O Arranjo agora guarda os sinais ao salvar (antes, salvar uma
   camada editada que usava `*nome` a soltava do sinal). Falta: o Link de um parâmetro que
   não é número (as fontes `trigger:`), e o knob na opacidade de um Generator (o motor só
   aceita número).
4. **FX numa faixa** (pixel_sort só no dançarino): a faixa vira um grupo que é composto à
   parte (a imagem dela + os seus FX, com o recorte) e só então misturado. É a mudança de
   motor (CPU e GPU). Antes dela, a saída barata é o FX valer só dentro da máscara do
   personagem (`mask:`, já existe).

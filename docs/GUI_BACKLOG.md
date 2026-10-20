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
   **Feito (16/10/2026):** `fx: [...]` numa camada (clipe, personagem, Generator, a
   montagem): os FX dela, em ordem, trabalham só na imagem dela (`core/video/track_fx.py`,
   uma pilha pequena com a imagem da faixa como base: os FX mantêm `bars:`, `mix:` e
   knobs). Um personagem vai para a cadeia sobre preto, com o seu alfa; o que os FX deixam
   fora da figura some, a não ser com `spill: true` (rastros, bordas, brilho: o que for
   claro aparece). Na timeline os FX da faixa aparecem embaixo dela ("> FX pixel_sort",
   "on lion only"), com compassos, plugin, bypass e Ctrl+Z; botão direito "Add FX to
   <faixa> only"; o inspector do personagem tem a caixa do spill. A cadeia mistura na CPU
   (um FX de shader nela ainda usa a GPU, a imagem indo e voltando a cada quadro).

## 15. [x] O Arranjo como duas linhas: BG e FG, e os FX em três lugares

**Problema (16/10/2026).** A timeline da fase 2 mostrava uma faixa por camada da cena: no
elenco, 29 faixas. A ideia era outra: duas linhas no tempo, fundo e frente, cada uma com a
sua lista de FX aplicados em sequência.

**Feito.** A janela guarda a cena como a montagem de clipes, a linha **BG**, a linha **FG** e a
**Master** (`core/arrangement.py`: `grouped`, `members_of`, `pack`):
- cada linha é uma camada `source: group` com `layers:` (os membros: um clipe nos seus
  blocos, um personagem, um Generator nos seus `bars:`) e `fx:`; o BG (`canvas: below`) é
  desenhado sobre a montagem e os FX dele pegam tudo; o FG é desenhado à parte, com alfa,
  e os FX dele pegam só os personagens (`core/video/track_fx.py`: `Group`; sem FX ele vai
  direto para a pilha, sem custo);
- coisas ao mesmo tempo numa linha empilham em sub-linhas (os cortes do REAPER pelos
  compassos em que tocam); a última sub-linha de cada linha fica vazia para pôr algo novo;
- escolher um clipe (Clips) ou um Generator (Generators) e arrastar numa linha o põe lá;
- os FX moram no **bloco** (só nele), na **linha** (nela toda) e na **Master** (sobre tudo):
  botão direito no bloco / na linha / em qualquer lugar; a lista (no inspector, clicando no
  nome da linha, num bloco ou na MASTER) tem liga/desliga, compassos ("9-12, 30-33"),
  plugin, ordem e apagar; os compassos dos FX da linha e da Master aparecem como traços finos;
- uma cena plana (como era o elenco) é agrupada ao abrir, pelo lugar de cada camada: abaixo
  do primeiro personagem é BG (os FX viram os FX do BG), até o último clipe acima dele é FG,
  acima é Master. Salvar grava assim; os comentários das camadas que viram membros se perdem.

**Pendências fechadas (16/10/2026):**
- `opacity:` de uma camada aceita um knob da música (motor e editor de camadas); no plugin
  de um Generator, a opacidade é um parâmetro com Link;
- o recorte com "tudo menos o fundo" (`drop:`) deixava uma linha do fundo na borda do
  quadro (pixels que o SAM2 não nomeou): os que tocam a borda agora contam como fundo;
- "Who plays" no plugin: um parâmetro de disparo (`trigger: {track: piano, envelope: 0.4}`)
  abre o editor no modo "quem toca", e o resto do disparo (o envelope) fica;
- zoom na régua: − / + / all, a barra "from bar", Ctrl+roda (no compasso sob o mouse) e
  Shift+roda (andar);
- "Preview bars" num bloco (ou no botão direito): o preview do studio daqueles compassos
  (até 8) com o arranjo como está, salvo ou não, numa janela própria.

## 16. [x] Tudo o que aparece no Arranjo se edita

Pedido (16/10/2026): "agora eu vejo o lofi mas não posso remover pra editar". Regra: o que
está desenhado é um bloco que se escolhe com um clique, se arrasta, se redimensiona pelas
pontas, se apaga com Delete e tem menu no botão direito (duplo clique: o plugin).

- **Trilhas de FX:** os FX do FG, do BG e do Master são blocos nas suas trilhas (FX que
  não se cruzam dividem a linha), não mais tracinhos. Arrastar um FX escolhido em FX
  numa trilha o põe ali; earlier/later mudam a ordem da cadeia.
- **CLIPS (a montagem):** mostra o clipe de cada compasso como ele toca (seed, ordem,
  cortes). Um clique, ou um arrasto, escolhe compassos. Daí:
  - "Put" fixa ali o clipe escolhido à esquerda;
  - "Nothing here" / Delete deixa vazio (o BG, ou preto);
  - "Back to the montage" volta à escolha da montagem;
  - "Make it a BG block" vira um bloco do BG.

  Os pins ficam na camada `clips` (`pins: {3: lofi_2, 9: none}`, compassos a partir de 1).
  Um compasso fixado não muda os clipes que a montagem sorteia para os outros, e um clipe
  fixado em compassos seguidos continua tocando em vez de recomeçar. Um clipe de outra
  pasta, ou de outro ponto, vai para o BG.
- **Nomes:** cada camada sem `name:` ganha o do seu comentário na cena (`# the lion` no fim
  da linha, ou a linha acima: "the bottles sing the voice"). Salvar grava `name:`, e o
  inspetor tem um campo para dar ou trocar o nome.
- **Compassos escolhidos e preview (16/10/2026):** "seleciono tantos bars na lane de bg,
  mando preview e nada acontece". Agora:
  - arrastar sobre os números dos compassos (a régua) escolhe compassos; numa lane, sem nada
    escolhido à esquerda para pôr, também. Escolher um bloco escolhe os compassos dele;
  - o botão "Preview bars a-b (Space)" na barra de cima, ou Espaço, toca esses compassos;
  - a janela do preview mostra o andamento, ou por que falhou (antes isso só aparecia na
    barra do Studio, atrás da janela), mostra o quadro em pé e tem Stop.

  O preview também começava a montagem do zero no compasso pedido, então mostrava outros
  clipes. Agora ele a reproduz desde o começo da música (sem decodificar), e os clipes
  batem com a linha CLIPS e o render.
- **O Preview nunca recusa calado (16/10/2026):** "clico em preview e nada acontece". As
  recusas só apareciam na linha verde da barra de cima, cortada na borda da janela. Agora a
  janela do preview sempre abre e diz o que acontece:
  - projeto sem Load: carrega e toca quando terminar;
  - o Studio ainda faz outro preview: estes compassos vêm em seguida;
  - nenhum compasso escolhido: toca os 8 primeiros à vista.

  A linha de status quebra a linha. A Arrange aberta antes do Load se refaz com os
  compassos reais da música quando ela carrega.
- **Botões mortos (17/10/2026):** o Preview do inspetor da linha CLIPS não fazia nada. O
  DearPyGui passa o `user_data` (None) como terceiro argumento posicional, então
  `lambda s, a, fn=fn: fn()` chamava `None()`, e o erro era engolido calado. Isso quebrava
  todos os botões e menus feitos em laço:
  - os botões da montagem e o menu do botão direito;
  - na lista de FX: on/off, plugin, ^, v e x;
  - no plugin: Who plays e x;
  - no Studio: as camadas, os gatilhos e os uniforms.

  Agora o valor capturado vem depois de `*` (keyword-only). Um teste impede que o padrão
  volte.

## 17. [x] Demos de cada Generator e FX
"Muito nome de efeito ou gerador não quer dizer nada até que você veja ele funcionando."

**O demo**
- Um vídeo de 512×512, ~10 s, 30 fps, pré-renderizado e **só visual, sem música**.
- Arquivos: `render_output/demos/<chave>.mp4`, mais um `.json` com a receita e um hash
  (spec, arquivo do shader ou da camada, fundo). Só o que mudou é re-renderizado.
- Ao escolher um Generator ou FX na lista, o demo toca em loop no painel embaixo dela.
  Clicar abre em 512. Sem demo: "no demo yet" e o botão Make demo.

**O movimento**
- Sem áudio, os knobs que a música moveria são varridos por `lfo:` numa grade sintética
  fixa, de ponta a ponta da faixa de cada um. Assim se vê o que o parâmetro faz.
- Uma faixa `meters` fina diz qual knob está variando.
- O render recebe um WAV mudo com a grade fixa: o `clip_generator` exige `--file`.
- Camadas que só se mexem com features de áudio (sem knob) ficam marcadas e ganham ajuste
  próprio.

**Generator:** tela cheia.

**FX:** metade sem / metade com. Esquerda sem o efeito, direita com, uma linha no meio. O
fundo é escolhido por categoria para o efeito saltar aos olhos:

| Categoria | Fundo |
|---|---|
| Lens & distortion | grade quadriculada com anel e texto em movimento |
| Time & trails | figura clara atravessando um fundo liso |
| Colour & light, Matter | um clipe colorido e contrastado da biblioteca, sempre o mesmo |
| Cuts | a montagem cortando a cada compasso |
| Figures | `me_diz_capa_base` e as máscaras SAM2 que já existem |

**Ajustes:** `demos.yaml`, só o que foge do padrão. 28 dos 66 entries têm specs afinadas
em `examples/me_diz_*.yaml`, de onde vêm as formas dos knobs (o áudio troca por LFO).

**Fases**
1. Receita, fundos, WAV mudo e grade, e o comando `python -m tools.demos render [--only]
   [--force]`, com 3 amostras: um Generator, um FX de lente, um FX de rastro.
2. O player no painel do catálogo: thread de decodificação (cv2), textura no `tick()`, para
   ao trocar de entry.
3. Lote dos 66 na GPU, em background e retomável (estimativa de 30–60 min; os fractais 3D
   são os lentos), folha de contato e ajuste dos fracos.
4. Testes:
   - toda entry tem receita válida e as necessidades atendidas;
   - o hash muda com o shader;
   - o player faz loop.

**Feito (17/10/2026):** `core/demos.py` e `core/demos.yaml`. Os 66 demos estão em
`render_output/demos/`, cerca de 120 MB, 2 a 30 s cada.

- **O que move o demo:** um loop sintético a 120 BPM, que ninguém ouve: bateria em
  "drums (midi)", acordes em "organ", "e-piano", "bass acid" e uma partitura dos acordes.
  As cenas `examples/me_diz_*` funcionam nele sem mudança, então 28 receitas vêm delas.
- **`core/demos.yaml`:** uma receita própria (`layer:`) ou ajustes sobre a afinada
  (`set:`, `drop:`). Hoje cobre:
  - meters e cast;
  - cubes, que precisa de uma pasta de clipes;
  - solid, que pisca no kick;
  - flame, sem pintar com o vídeo de baixo;
  - rgb_noise, mais forte;
  - shockwave e drops, que só agem com notas;
  - pixel_sort, varrido de 0 a 1;
  - kifs_acorde, com o fatiamento mais estreito;
  - sombras_2d_kicks, com os knobs ligados;
  - mouths, que abrem com os médios em vez da voz;
  - time_map, que usa as figuras.

  Camadas que seguem figuras (`figure:`, `map: figures`) vão sozinhas para o fundo com
  máscaras.
- **Comandos:** `python -m core.demos render` (só os que faltam ou ficaram velhos),
  `list` e `sheet` (folha de contato).
- **Na GUI:** o demo toca embaixo da lista de Generators e FX; clicar abre em 512. "no demo
  yet" e o demo velho têm botão para refazer, em background.

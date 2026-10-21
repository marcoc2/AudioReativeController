# Exemplos — cenas prontas do ARC

Cada `.yaml` é uma cena completa; o `.md` ao lado traz o comando CLI e o
passo a passo pra replicar na GUI (`arc_studio.py`).

| Cena | O que faz |
|------|-----------|
| [`deserto_kick_reverse`](deserto_kick_reverse.md) | Clipe por compasso (5/4), bumbo inverte o playback + gravity warp |
| [`enxame_clips_handover`](enxame_clips_handover.md) | Bumbo troca clipes até a caixa gravada entrar e assumir (`until`/`exclude`) |
| [`enxame_cells`](enxame_cells.md) | Células 8-bit sobre os clipes; harmonia pinta, caixa dispara mitose |
| [`enxame_julia_solo`](enxame_julia_solo.md) | Julia 8-bit navegando a cardioide; kick = zoom, caixa = inversão |
| [`enxame_voo`](enxame_voo.md) | Voo infinito por corredor fractal 3D (GPU); kick = surto de velocidade |
| [`enxame_cubos`](enxame_cubos.md) | Palco 3D (GPU): clipes reativos nas faces de cubos que a câmera atravessa; kick inverte/avança |
| [`enxame_orbiters`](enxame_orbiters.md) | Mandala de órbitas aninhadas e satélites; centroid/flux guiam raios |
| [`enxame_particles`](enxame_particles.md) | Partículas em órbita; batidas de caixa disparam jatos de faíscas nos vértices |
| [`enxame_feedback`](enxame_feedback.md) | Túnel psicodélico de feedback com zoom reativo a bumbo e rotação espectral |
| [`enxame_echoes`](enxame_echoes.md) | Ecos de vídeo dinâmicos com redução progressiva de escala (Doppler) e hue shift |

## Cenas com as figuras do vídeo (máscaras SAM2)

Estas cenas precisam da trilha de máscaras do render base. O comando de três
passos está no cabeçalho de cada `.yaml`: render só com os clipes, depois
`segment_video.py`, depois a cena com `--masks`. Os atos estão descritos em
`core/video/cast.py` e em `docs/EFFECTS_BACKLOG.md` (família 21–24).

| Cena | O que faz |
|------|-----------|
| `esfolado_world_melts` | `mask: outside`: atrás das figuras o papel vira o túnel de octagramas; só elas levam a aberração da caixa |
| `esfolado_cast` | Elenco: bumbo = salto, caixa = tranco, chimbal = halo, baixo tinge o fundo |
| `esfolado_melodia` | Melodia no corpo: `echo` (fantasmas enquanto a nota soa) e `rings` (sonar no contorno) |
| `esfolado_colagem` | `carry`: as figuras atravessam o corte como adesivo até a próxima batida |
| `esfolado_completo` | Tudo junto: túnel de octagramas fora, aberração dentro, sete papéis |
| `esfolado_rastro` | `wake`: rastro temporal ligado pela nota (`hold: note`) e pelo compasso (`hold: bar`) |
| `esfolado_rastro_longo` | O mesmo com rastro 4× mais longo e carimbado a cada semicolcheia (`every: 1/16`) |
| `esfolado_rastro_deriva` | O longo com o rastro derivando numa direção nova a cada clipe (`shift`) |
| `carrossel_completo` | Os efeitos do `esfolado_completo` orquestrados numa valsa (3/4), clipe a cada 2 compassos |
| `carrossel_retrato` | Retrato 480×854 sobre visualizers (clipes largos girados 90°, `clip_end: loop`) |
| `carrossel_abstrato` | Retrato sobre clipes abstratos quadrados cortados (`--clip-fit cover`) |
| `me_diz_capa` | A capa do single virou seis movimentos de câmera; a cena do carrossel sobre ela |

As cenas `me_diz_*` abaixo são demonstrações de um efeito só: cada uma tem o efeito sozinho
sobre os clipes (ou em tela cheia, quando ele gera a imagem) e, por cima, a camada `meters`:
um título no alto e, embaixo, uma faixa que rola a música pela agulha (um traço por knob, o
piano-roll das notas, os acordes, as linhas de compasso e os cortes em vermelho). Elas
começam no compasso 18.3 da DAW (35.993 s do master), onde só tocam `drums (midi)` e `organ`.

| Cena | O que faz |
|------|-----------|
| `me_diz_acorde` | Knobs `chord`/`cut`: o túnel muda de forma a cada corte; cinza em acorde menor, colorido em maior |
| `me_diz_paleta` | `chord_color`: o acorde que soa recolore o quadro (rampa das notas do acorde; maior claro, menor escuro) |
| `me_diz_transicao` | `transition`: o clipe novo entra pelas figuras, por varredura, por manchas ou do centro |
| `me_diz_lente` | `flame_lens`: o clipe passa pelas variações do flame (swirl na caixa, bubble no órgão), liso entre as batidas |
| `me_diz_kifs` | O caleidoscópio 3D do acorde (KIFS): octaédrico no menor, icosaédrico no maior |
| `me_diz_newton` | Fractal de Newton cujas raízes são as notas do acorde; cada bacia mostra o clipe girado pela sua nota |
| `me_diz_lyapunov` | A levada do bumbo em cada compasso desenha uma paisagem de Lyapunov (knob `groove`) |
| `me_diz_julia` | Julia cuja armadilha de órbita são as figuras (`channel: mask_distance`): elas se repetem pelo fractal |
| `me_diz_droste` | O quadro dentro de si mesmo na espiral de Escher, um nível por compasso |
| `me_diz_datamosh` | `datamosh`: no corte, o plano velho é arrastado pelo movimento do novo, em blocos |
| `me_diz_envelopes` | Knobs `adsr` (órgão), `relative` (grave contra a média recente) e `before` (antes da caixa) |
| `me_diz_lfo` | Knobs `lfo` e `random_on` dirigindo a lente do flame |
| `me_diz_marmore` | Domain warp: o mundo derrete como mármore (grave e presença relativos) |
| `me_diz_fluxo` | Fluxo cromático (modulateHue da Hydra): as cores escorrem como tinta |
| `me_diz_hiperbolico` | *Circle Limit*: o quadro ladrilhado no disco de Poincaré, um ladrilho por tempo |
| `me_diz_copiadora` | MRCM: quatro lentes de cópias, uma por instrumento |
| `me_diz_perolas` | Pérolas de Indra com o clipe dentro; a nota do órgão muda o tamanho das pérolas |
| `me_diz_milkdrop` | Campo de warp do MilkDrop com o vórtice na maior figura (knob `figure`) |
| `me_diz_kifs_video` | O KIFS do acorde pintado com o clipe (projeção triplanar) |
| `me_diz_fatiador` | O KIFS cortado por um plano que atravessa uma vez por compasso |
| `me_diz_nebulosa` | Nebulosa volumétrica (kaliset); o kick acelera a viagem |
| `me_diz_fases` | Um fractal de dobras que muda de forma a cada corte |
| `me_diz_bolhas` | Espuma de bolhas apolonianas; os acordes do órgão acendem as cascas |
| `me_diz_reacao` | Gray-Scott crescendo das bordas; o modo do acorde escolhe o regime |
| `me_diz_reacao_figuras` | O mesmo coral crescendo só nas figuras (`mask: inside`), semeado pelas notas do órgão |
| `me_diz_tempo` | `time_map`: as figuras no agora, o mundo atrasando em semicolcheias |
| `me_diz_pixelsort` | `pixel_sort`: as faixas crescem enquanto o órgão soa |
| `me_diz_bloom` | Bloom: o kick baixa o limiar, os pratos sobem o ganho |
| `me_diz_canto` | Atos `sing` (a maior figura sobe com o órgão) e `clock` (a segunda só se mexe no bumbo) |
| `me_diz_scope` | Ato `scope`: osciloscópio no contorno com os acordes do órgão |
| `me_diz_portais` | Transições `menger` e `julia` |
| `me_diz_espiral` | Transição `spiral` |
| `me_diz_julia4d` | Julia quaterniônico: o acorde move c, a nota do órgão percorre a 4ª dimensão |
| `me_diz_luzes` | As notas do órgão como luzes dentro do KIFS (knob `voice`) |
| `me_diz_hibrido` | A levada escolhe a fórmula de cada iteração de um fractal 3D |
| `me_diz_desfoque` | Desfoque de movimento no túnel (`subframes`) |
| `me_diz_nova` | Newton em modo Nova, girando com o baixo do órgão |
| `me_diz_chama` | Mandala de flame com um xform por instrumento (`weights`) |
| `me_diz_chama_suave` | A mesma, com a estimativa de densidade do flam3 (`estimator`) |
| `me_diz_nebulabrot` | `nebulabrot`: as órbitas que escapam, cada canal numa banda |
| `me_diz_mergulho` | `deep_zoom`: um mergulho de onze décadas no Mandelbrot (perturbação) |
| `me_diz_voo` | `flythrough`: um voo pelos corredores do pseudo-kleiniano, um passo por compasso |
| `me_diz_cenario` | As figuras em pé num salão fractal, com a sombra dele sobre elas |

Opções de clipe usadas nessas cenas:
- `bars_per_clip: N`: troca de clipe a cada N compassos;
- `clip_order: shuffle`: embaralha os clipes;
- `clip_end: loop`: o clipe curto recomeça em vez de voltar de trás pra frente;
- `--clip-turn-wide` (CLI): gira em 90° os clipes largos, para render retrato.

O render grava `<stem>_cuts.json`, e o `segment_video.py` usa esses cortes em vez
de adivinhá-los pela imagem.

## Fontes de camada

Disponíveis (combináveis nas cenas acima), lista em `LAYER_SOURCES`
(`core/video/layers.py`):
- `solid` (flash por trigger), `cells` (células 8-bit com mitose), `mandelbulb` (fractal 3D orbital em GPU);
- `shadertoy` (shaders do Shadertoy, ver `shaders/shadertoy/README.md`), `rgb_split` (aberração cromática na batida);
- `cast` (o elenco sobre as máscaras);
- `chord_color` (o acorde recolore o quadro), `transition` (o corte como gesto), `flame_lens` (o clipe pelas variações do flame).
- `datamosh` (o corte como glitch de codec), `time_map` (cada pixel do seu momento), `pixel_sort` (faixas de pixels ordenados);
- `nebulabrot`, `deep_zoom` (mergulho no Mandelbrot), `flythrough` (voo por um fractal);
- `meters` (a faixa que mostra o que move a cena: knobs, notas, acordes, compassos e cortes; para avaliar um efeito).

Blend modes: `normal`, `add`, `screen`, `multiply`. `opacity` pode ser estática ou
por envelope de trigger. `mask:` (dentro/fora das figuras) vale em qualquer camada.

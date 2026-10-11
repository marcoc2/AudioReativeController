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
- `cast` (o elenco sobre as máscaras).

Blend modes: `normal`, `add`, `screen`, `multiply`. `opacity` pode ser estática ou
por envelope de trigger. `mask:` (dentro/fora das figuras) vale em qualquer camada.

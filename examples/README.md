# Exemplos — cenas prontas do ARC

Cada `.yaml` é uma cena completa; o `.md` ao lado traz o comando CLI e o
passo a passo pra replicar na GUI (`arc_studio.py`).

| Cena | O que faz |
|------|-----------|
| [`deserto_kick_reverse`](deserto_kick_reverse.md) | Clipe por compasso (5/4), bumbo inverte o playback + gravity warp |
| [`enxame_clips_handover`](enxame_clips_handover.md) | Bumbo troca clipes até a caixa gravada entrar e assumir (`until`/`exclude`) |
| [`enxame_julia_solo`](enxame_julia_solo.md) | Julia 8-bit navegando a cardioide; kick = zoom, caixa = inversão |
| [`enxame_voo`](enxame_voo.md) | Voo infinito por corredor fractal 3D (GPU); kick = surto de velocidade |

Outras fontes de camada disponíveis (combináveis nas cenas acima):
`solid` (flash por trigger), `cells` (células 8-bit com mitose),
`mandelbulb` (fractal 3D orbital em GPU). Blend modes: `normal`, `add`,
`screen`, `multiply`; `opacity` estática ou por envelope de trigger.

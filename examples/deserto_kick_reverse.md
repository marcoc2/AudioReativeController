# Deserto de Lama — clipe por compasso + bumbo reverte (`deserto_kick_reverse.yaml`)

O cenário original do compositor de clipes: cada compasso (5/4, lido do
MIDI) ganha um miniclipe novo; **todo bumbo inverte a direção do
playback** (ping-pong nas bordas), e o bloco `gravity` faz cada bumbo
agir como poço gravitacional — o playback acelera chegando no hit e
desacelera saindo (peak/floor/radius/curve).

## CLI

```powershell
.venv\Scripts\python.exe clip_generator.py `
  --file input/deserto_de_lama.mp3 `
  --midi input/deserto_de_lama_ep_clean.mid `
  --clips C:\Users\marco\Videos\deserto_square `
  --scene examples/deserto_kick_reverse.yaml `
  --bars 0 --cache-size 101 --resolution 480x480 --clip-order shuffle `
  --output render_output/deserto_full.mp4
```

Overrides rápidos sem editar YAML: `--gravity-peak 7 --gravity-floor 0.7
--gravity-curve 3` (a config do render que ficou melhor nos testes).

## Na GUI (ARC Studio)

1. **Audio** = `input/deserto_de_lama.mp3`, **MIDI** =
   `input/deserto_de_lama_ep_clean.mid`, **Scene** =
   `examples/deserto_kick_reverse.yaml`.
2. **Mode: clips** → **Clips Folder** = `deserto_square`; Order =
   shuffle; Full song se quiser a música toda. **Load Project** (a barra
   mostra `BPM 130.0  5/4`).
3. Os knobs de **Gravity** no grupo clips (checkbox + peak/floor/radius/
   curve) sobrescrevem a cena pra experimentar; a lane **speed** na
   timeline mostra a curva resultante sem renderizar.
4. **Preview** com X bars pra ouvir/ver; **Render Full**.

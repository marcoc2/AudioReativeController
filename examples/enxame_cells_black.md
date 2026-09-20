# Enxame — Células 8-bit no microscópio com Fundo Preto (`enxame_cells_black.yaml`)

O gerador "psicodélico de 8 bits": células como vistas no microscópio,
numa grade minúscula posterizada com upscale nearest (pixel art
autêntica), mas renderizadas diretamente sobre um fundo preto (sem footage/clipes).
Mapeamento de áudio: **chroma** amarra cada célula a uma
classe de altura — a harmonia literalmente pinta a lâmina; **flux**
ondula as membranas; **graves** aceleram a deriva browniana; e cada
batida da **caixa gravada** (onsets do stem) dispara uma **mitose** — a
população cresce com a música (até `n_max`).

## CLI

```powershell
.venv\Scripts\python.exe clip_generator.py `
  --file "C:\Audio\recordings\projeto2024\ep\stems\enxame\mix_enxame_ep_15_07_2026.mp3" `
  --midi "C:\Audio\recordings\projeto2024\ep\stems\enxame\enxame_drum_no_snare.mid" `
  --scene examples/enxame_cells_black.yaml `
  --bars 6 --resolution 480x480 --start-time 42 `
  --output render_output/enxame_cells_black.mp4
```

Versão retrato pra celular: `--resolution 2160x3840 --fps 60` (as células continuam quadradas; a grade estica em blocos inteiros).

## Na GUI (ARC Studio)

1. **Audio** = mix do Enxame, **MIDI** = `enxame_drum_no_snare.mid`,
   **Scene** = `examples/enxame_cells_black.yaml`.
2. **Skip AI separation** + **Load Project** — a timeline mostra a stem da caixa como waveform se estiver no painel STEMS.
3. **Preview** pra ver as células pulsando sobre fundo preto;
   **Render Full** (Enc: `nvenc` pra resoluções altas).

Knobs: `resolution` (grão do pixel art — 80 fica bem retrô, 160 mais
fino), `n_base`/`n_max` (dramaturgia populacional), `seed` (layout inicial das células).

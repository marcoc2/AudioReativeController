# Enxame — Células 8-bit no microscópio (`enxame_cells.yaml`)

O gerador "psicodélico de 8 bits": células como vistas no microscópio,
numa grade minúscula posterizada com upscale nearest (pixel art
autêntica). Mapeamento de áudio: **chroma** amarra cada célula a uma
classe de altura — a harmonia literalmente pinta a lâmina; **flux**
ondula as membranas; **graves** aceleram a deriva browniana; e cada
batida da **caixa gravada** (onsets do stem) dispara uma **mitose** — a
população cresce com a música (até `n_max`). Nesta cena as células são
compostas em `screen` sobre os miniclipes; remova a camada `clips` para
tê-las sozinhas em fundo preto.

## CLI

```powershell
.venv\Scripts\python.exe clip_generator.py `
  --file "C:\Audio\recordings\projeto2024\ep\stems\enxame\mix_enxame_ep_15_07_2026.mp3" `
  --midi "C:\Audio\recordings\projeto2024\ep\stems\enxame\enxame_drum_no_snare.mid" `
  --clips <SUA_PASTA_DE_CLIPES> `
  --scene examples/enxame_cells.yaml `
  --bars 6 --resolution 480x480 --start-time 42 `
  --output render_output/enxame_cells.mp4
```

Versão retrato pra celular: `--resolution 2160x3840 --fps 60 --codec nvenc`
(as células continuam quadradas; a grade estica em blocos inteiros).

## Na GUI (ARC Studio)

1. **Audio** = mix do Enxame, **MIDI** = `enxame_drum_no_snare.mid`,
   **Scene** = `examples/enxame_cells.yaml`.
2. **Mode: clips** → **Clips Folder** = sua pasta de miniclipes
   (obrigatória porque a cena tem a camada `clips`; sem ela, delete a
   camada no YAML e a pasta deixa de ser exigida).
3. **Skip AI separation** + **Load Project** — a timeline mostra a lane
   TRIGGERS com o handover, e a stem da caixa como waveform se estiver
   no painel STEMS.
4. **Preview** pra ver as células pulsando sobre o footage;
   **Render Full** (Enc: `nvenc` pra resoluções altas).

Knobs: `resolution` (grão do pixel art — 80 fica bem retrô, 160 mais
fino), `n_base`/`n_max` (dramaturgia populacional), `blend`
(`add`/`screen`), `seed` (layout inicial das células).

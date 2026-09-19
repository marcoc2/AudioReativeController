# Enxame — Partículas em Vértices (`enxame_particles.yaml`)

O gerador de física de partículas 8-bit. As partículas flutuam sob gravidade interna em órbita do centro e variam de matiz conforme o centroide espectral. A batida da **caixa gravada** (onsets no stem de áudio) dispara um jato volumoso de partículas (*burst*) disparadas radialmente a partir de cada um dos vértices do pentágono rotativo central.

## CLI

```powershell
.venv\Scripts\python.exe clip_generator.py `
  --file "C:\Audio\recordings\projeto2024\ep\stems\enxame\mix_enxame_ep_15_07_2026.mp3" `
  --midi "C:\Audio\recordings\projeto2024\ep\stems\enxame\enxame_drum_no_snare.mid" `
  --scene examples/enxame_particles.yaml `
  --bars 8 --resolution 480x480 --start-time 42 `
  --output render_output/enxame_particles.mp4
```

## Na GUI (ARC Studio)

1. **Audio** = mix do Enxame, **MIDI** = `enxame_drum_no_snare.mid`,
   **Scene** = `examples/enxame_particles.yaml`.
2. O stem de áudio da caixa (`enxame_ep_15_07_2026-005_snare.mp3`) é carregado pelo compositor para ler os onsets automaticamente.
3. **Skip AI separation** + **Load Project**.
4. **Preview** para ver a explosão rítmica de faíscas nos vértices do pentágono;
   **Render Full**.

Knobs no YAML: `n_particles` (tamanho total do pool de física), `n_vertices` (número de pontas do polígono emissor), `quantity` (partículas por explosão).

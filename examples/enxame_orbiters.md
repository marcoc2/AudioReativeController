# Enxame — Orbitadores em Cascata (`enxame_orbiters.yaml`)

O gerador geométrico de mandalas orbitantes em cascata. O centroide espectral (brilho) estica a órbita principal; o fluxo de áudio expande os pequenos satélites externos; a velocidade de órbita do mandala responde ao grid rítmico (compassos e batidas); e cada ramo adota a cor e intensidade dinâmica de sua classe harmônica correspondente via **chroma**.

## CLI

```powershell
.venv\Scripts\python.exe clip_generator.py `
  --file "C:\Audio\recordings\projeto2024\ep\stems\enxame\mix_enxame_ep_15_07_2026.mp3" `
  --midi "C:\Audio\recordings\projeto2024\ep\stems\enxame\enxame_drum_no_snare.mid" `
  --scene examples/enxame_orbiters.yaml `
  --bars 8 --resolution 480x480 --start-time 42 `
  --output render_output/enxame_orbiters.mp4
```

## Na GUI (ARC Studio)

1. **Audio** = mix do Enxame, **MIDI** = `enxame_drum_no_snare.mid`,
   **Scene** = `examples/enxame_orbiters.yaml`.
2. **Skip AI separation** + **Load Project**.
3. **Preview** pra assistir ao relógio planetário em tempo real;
   **Render Full**.

Knobs no YAML: `resolution` (pixel art grão), `n_parents` (braços principais), `n_satellites` (satélites orbitantes).

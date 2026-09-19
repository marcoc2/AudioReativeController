# Enxame — Ecos de Rastro no Fractal (`enxame_echoes.yaml`)

O efeito de ecos de rastro dinâmico (Doppler + Hue Shift) aplicado como post-op sobre o conjunto fractal de Julia. À medida que o fractal morfa e reage às frequências, as imagens passadas são deixadas para trás encolhendo gradualmente (Doppler) e rotacionando suas cores (RGB channel shifting) sincronizadas à velocidade do fluxo de transição espectral (`flux`).

## CLI

```powershell
.venv\Scripts\python.exe clip_generator.py `
  --file "C:\Audio\recordings\projeto2024\ep\stems\enxame\mix_enxame_ep_15_07_2026.mp3" `
  --midi "C:\Audio\recordings\projeto2024\ep\stems\enxame\enxame_drum_no_snare.mid" `
  --scene examples/enxame_echoes.yaml `
  --bars 8 --resolution 480x480 --start-time 42 `
  --output render_output/enxame_echoes.mp4
```

## Na GUI (ARC Studio)

1. **Audio** = mix do Enxame, **MIDI** = `enxame_drum_no_snare.mid`,
   **Scene** = `examples/enxame_echoes.yaml`.
2. **Skip AI separation** + **Load Project**.
3. **Preview** para ver os clones fractais se afastando em cascata cromática;
   **Render Full**.

Knobs no YAML: `depth` (quantidade de frames de eco mantidos na memória), `opacity_decay` (desvanecimento), `scale_decay` (fator Doppler de redução de tamanho), `hue_shift` (velocidade de alteração das cores).

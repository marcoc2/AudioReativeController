# Enxame — Feedback de Células (`enxame_feedback.yaml`)

O efeito de feedback acumulativo (túnel psicodélico) aplicado como post-op sobre o gerador de células de microscópio. O bumbo (MIDI) aciona um pulso de zoom no túnel de feedback, enquanto a rotação responde de forma amortecida (lowpass) ao centroide espectral (brilho espectral).

## CLI

```powershell
.venv\Scripts\python.exe clip_generator.py `
  --file "C:\Audio\recordings\projeto2024\ep\stems\enxame\mix_enxame_ep_15_07_2026.mp3" `
  --midi "C:\Audio\recordings\projeto2024\ep\stems\enxame\enxame_drum_no_snare.mid" `
  --scene examples/enxame_feedback.yaml `
  --bars 8 --resolution 480x480 --start-time 42 `
  --output render_output/enxame_feedback.mp4
```

## Na GUI (ARC Studio)

1. **Audio** = mix do Enxame, **MIDI** = `enxame_drum_no_snare.mid`,
   **Scene** = `examples/enxame_feedback.yaml`.
2. **Skip AI separation** + **Load Project**.
3. **Preview** para ver os rastros circulares de células rotacionando e se afundando num túnel reativo;
   **Render Full**.

Knobs no YAML: `decay` (velocidade com que a imagem anterior desvanece), `base_scale`/`max_scale` (tamanho e amplitude da pulsação do túnel), `max_rotate` (ângulo de rotação máximo).

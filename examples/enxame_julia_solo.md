# Enxame — Julia solo (`enxame_julia_solo.yaml`)

Conjunto de Julia 2D morfando pela borda do Mandelbrot, posterizado
8-bit, fundo escuro. A harmonia (chroma) desliza o parâmetro `c` pela
cardioide; a energia grave atravessa a fronteira do caos; o bumbo (MIDI)
pulsa zoom e a caixa **gravada em áudio** (onsets detectados no stem)
inverte a paleta por 100ms. CPU puro, determinístico.

## CLI

```powershell
# quadrado (como os testes):
.venv\Scripts\python.exe clip_generator.py `
  --file "C:\Audio\recordings\projeto2024\ep\stems\enxame\mix_enxame_ep_15_07_2026.mp3" `
  --midi "C:\Audio\recordings\projeto2024\ep\stems\enxame\enxame_drum_no_snare.mid" `
  --scene examples/enxame_julia_solo.yaml `
  --bars 16 --resolution 2160x2160 --start-time 38 `
  --output render_output/enxame_julia_solo_4k.mp4

# retrato pra celular, 60fps (o plano complexo se adapta ao aspecto):
#   --fps 60 --resolution 2160x3840
```

## Na GUI (ARC Studio)

1. **Audio** = mix do Enxame, **MIDI** = `enxame_drum_no_snare.mid`,
   **Scene** = `examples/enxame_julia_solo.yaml`.
2. **Mode: clips**, **Skip AI separation**, **Load Project**.
3. **Resolution**: `2160x2160` (ou `2160x3840` retrato) + Enter;
   **FPS**: 60 se quiser high-frame-rate.
4. **Preview** mostra o fractal real com os triggers; **Render Full**.

Knobs: `resolution` interna (grão do 8-bit), `iters` (detalhe da borda),
`threshold`/`min_gap` da detecção de onsets da caixa, envelopes.

# Enxame — Voo fractal infinito (`enxame_voo.yaml`)

Câmera voando pra frente por um corredor fractal periódico (KIFS de
Sierpinski espelhado nas paredes), com paralaxe real e névoa de
profundidade. O mundo é espaço-periódico, então o voo é infinito e o
vídeo loopa perfeitamente a cada `loop_bars`. Mapeamento de áudio: bumbo
(MIDI 36) = surto de velocidade; flux = brilho/energia da luz; centroide
= matiz base; as bandas de cor fluem junto com o voo.

## CLI

```powershell
.venv\Scripts\python.exe clip_generator.py `
  --file "C:\Audio\recordings\projeto2024\ep\stems\enxame\mix_enxame_ep_15_07_2026.mp3" `
  --midi "C:\Audio\recordings\projeto2024\ep\stems\enxame\enxame_drum_no_snare.mid" `
  --scene examples/enxame_voo.yaml `
  --bars 4 --resolution 3840x2160 --start-time 34.2857 `
  --output render_output/enxame_voo_4k.mp4
```

`--start-time` num múltiplo do período (compasso 16 = 34.2857s) faz o
mp4 fechar em loop perfeito. Cena 100% generativa: não precisa de
`--clips`.

## Na GUI (ARC Studio)

1. `python arc_studio.py` → botões **Audio** (mix do Enxame) e **MIDI**
   (`enxame_drum_no_snare.mid`).
2. Botão **Scene** → `examples/enxame_voo.yaml`.
3. **Mode: clips** (o modo cobre todo o pipeline de cenas), marque
   **Skip AI separation** e clique **Load Project**.
4. Resolution: digite `3840x2160` + Enter (ou preset). FPS/Bars a gosto.
5. **Render Full**. (O recorte `--start-time` ainda é só CLI.)

Knobs no YAML: `loop_bars` (velocidade da queda), `hue_spread` (riqueza
de cor), `envelope` do pulso. No motor (`core/mandelbox.py`): `clearance`
e `density` mudam o quão claustrofóbico é o túnel.

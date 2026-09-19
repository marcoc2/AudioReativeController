# Cubos no vazio — clipes nas faces, câmera atravessando (`enxame_cubos.yaml`)

Um **palco 3D**: os clipes tocam nas **faces de cubos** que flutuam num
espaço vazio (preto configurável), e a câmera avança em linha reta pelo meio
deles — no espírito do [`enxame_voo`](enxame_voo.md), só que o conteúdo é o
**próprio material** (os clipes), não um fractal. A geometria é cenografia;
**o que reage ao áudio são os clipes**, via `clip_reactivity` (uma config de
`ClipComposer` embutida na camada). Bumbo inverte o playback do clipe, caixa
troca de clipe, e cada bumbo também dá um surto de avanço na câmera
(`zoom_pulse`).

O corredor de cubos é **periódico**: a câmera avança exatamente 1 corredor a
cada `loop_bars` compassos, então o voo faz loop perfeito. Os cubos ficam num
anel (`radius_min`/`radius_max`) — o centro fica vazio, a câmera passa pelo
buraco.

Cena **autocontida**: a camada carrega o próprio pool de clipes (`clips_dir`),
então `--clips` na CLI **não** é usado por ela. Passe `--midi` para ter grade
musical + triggers de bateria.

## CLI

```powershell
python clip_generator.py `
  --file input/deserto_de_lama/deserto_de_lama.mp3 `
  --midi input/deserto_de_lama/deserto_de_lama_ep.mid `
  --scene examples/enxame_cubos.yaml `
  --bars 8 --resolution 854x480 `
  --output render_output/enxame_cubos.mp4
```

Retrato pra celular: `--resolution 1080x1920`. Resoluções altas: `--codec nvenc`.

## Na GUI (ARC Studio)

1. **Audio** = sua mix, **MIDI** = a bateria (bumbo nota 36, caixa 38/40),
   **Scene** = `examples/enxame_cubos.yaml`.
2. **Skip AI separation** + **Load Project**.
3. **Preview** pra ver os cubos passando; **Render Full** (Enc: `nvenc` em 4K).

Como o `clips_dir` está fixo na cena, a pasta de clipes da GUI não afeta esta
camada — edite `clips_dir` no `.yaml` para trocar o pool.

## Knobs

- `clips_dir` — pasta de mp4s mostrados nas faces (pool próprio da camada).
- `bg_color` — cor do espaço vazio (comece `[0,0,0]`).
- `n_cubes` — quantidade de cubos no corredor.
- `faces_with_clips` — 0..6 faces mostrando a tela de clipes; o resto usa `face_color`.
- `face_resolution` — resolução da textura por face (256 pra preview; 512 pra 4K nítido).
- `crop_faces` — `true` faz crop central dos clipes 16:9 pra preencher a face quadrada
  (sem faixas pretas); `false` mantém o clipe inteiro com letterbox.
- `radius_min`/`radius_max` — raio do anel de cubos (largura do "buraco" central).
- `loop_bars` — compassos por corredor (velocidade da câmera + período do loop).
- `zoom_pulse` — trigger de surto de avanço (MIDI `notes:` ou `audio:` de stem).
- `one_clip_per_face` — `false`: a **mesma** tela reativa em todas as faces;
  `true`: um `ClipComposer` **independente por face-index** (0..`faces_with_clips`-1),
  cada face um clipe reagindo por conta própria.
- `clip_reactivity` — bloco de `ClipComposer` (a tela reativa): `clip_per_bar`,
  `clip_order`, `seed`, `triggers` (`reverse`/`next_clip`/`random_clip`/`restart`,
  `gravity`, `until`, `exclude`) — tudo que uma cena de clipes normal aceita.

> **Memória:** com `one_clip_per_face`, os N clipes das faces compartilham uma
> única biblioteca de decode (custo O(faces), não O(cubos)). A variedade entre
> faces vem de seeds por face — use `clip_order: shuffle` ou `random`
> (`sequential` mantém as faces em sincronia de propósito). Requer GPU
> (moderngl), como o `enxame_voo`.

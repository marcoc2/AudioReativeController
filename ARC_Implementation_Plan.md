# ARC — Plano de Implementação

> Rastreamento de progresso do `ARC_Backlog.md`.
> Marque `[x]` quando concluído. Referência ao backlog entre parênteses.

---

## Sprint 1 — Fundação (sequencial, bloqueante)

- [x] **1.1** `core/rhythm/grid.py` — classe `RhythmGrid` (BPM, fórmula de compasso, fases, subdivisões)
- [x] **1.1.t** `tests/test_grid.py` — cobrir `phase`, `beat_phase`, `is_beat`, `is_downbeat`, subdivisões (15 testes, todos passando)
- [ ] **1.2** `core/rhythm/analyzer.py` — `analyze(y, sr) -> RhythmGrid` via `librosa.beat.beat_track`
- [ ] **1.2.c** Cache por `file_hash` em `rhythm_cache/<hash>.json`
- [ ] **R1** Refactor `VisualObject` — loop produz lista de objetos, não desenha inline (pré-requisito de 4.x/5.x, feito junto do 1.3)
- [ ] **1.3** Refactor loop de `animation_generator.py` — usa `grid.phase()` e `grid.is_downbeat()`
- [ ] **1.4** CLI `--bars N` — renderiza N compassos em vez de N segundos
- [ ] **1.5** CLI `--subdivision {quarter,eighth,16th,...}` — resolução mínima do grid

---

## Sprint 2 — Paralelo (desbloqueado após Sprint 1)

### Épico 2 — MIDI

- [ ] **2.1** `core/rhythm/midi_reader.py` — lê MIDI via `mido`, extrai BPM/timestamps/velocity; kick (C2) → downbeat
- [ ] **2.2** CLI `--midi path.mid` — substitui analyzer automático
- [ ] **2.3** Mapeamento canal MIDI → objeto visual (`midi_map.yaml`)
- [ ] **2.4** Velocity → intensidade visual (raio do objeto)
- [ ] **Deps** adicionar `mido>=1.3` ao `requirements.txt` (pular `pretty_midi` — redundante)

### Épico 3 — Features de áudio (não depende de nada)

- [ ] **3.1** Spectral Centroid — `features['centroid']`
- [ ] **3.2** Chroma Features — `features['chroma']` (12-vec) + `dominant_pitch`
- [ ] **3.3** Spectral Flux — `features['flux']`
- [ ] **3.4** Onset Detection — `features['onset']: bool` via `np.searchsorted`
- [ ] **3.5** Energia por sub-banda (sub-bass, bass, low-mid, mid, high-mid, presence, brilliance) — `features['subbands']`
- [ ] **Util** `core/cache.py` unificando file_hash de stems + rhythm (refactor oportunista)

---

## Sprint 3 — Expansão

### Épico 4 — Outputs

- [ ] **4.0** `core/output/base.py` — `Renderer` ABC (`init`, `render_frame`, `finalize`)
- [ ] **4.0b** `core/output/video.py` — mover render atual para cá
- [ ] **4.1** `core/output/sam_export.py` — JSON `{frame, t, bar_phase, objects: [{id,x,y,r}]}`
- [ ] **4.2** `core/output/depth_map.py` — gradiente radial para ControlNet Depth
- [ ] **4.3** `core/output/flow_map.py` — HSV direção/magnitude para AnimateDiff
- [ ] **4.4** CLI `--outputs video,depth,sam,flow` — múltiplos renderers em paralelo

### Épico 5 — Primitivos

- [ ] **5.1** Polígonos mutantes — N vértices por frequência dominante
- [ ] **5.2** `VectorField` em `core/motion.py` — perlin + perturbação por flux
- [ ] **5.3** Multi-objeto por stem — um `VisualObject` por stem ativo

---

## Sprint 4 — Complementos

- [ ] **3.6** Segment Detection — `core/rhythm/segments.py` via `librosa.segment.agglomerative` (usa grid maduro)
- [ ] **5.4** L-Systems rítmicos — `core/primitives/lsystem.py`, iteração por `grid.is_beat()`
- [ ] **5.5** Fractais de áudio — `core/primitives/fractal.py` (Julia CPU, 256² + upscale)

---

## Sprint 5 — Pose (exploratório, opcional)

- [ ] **6.1** `core/pose/extractor.py` — MediaPipe para extrair keypoints de vídeo
- [ ] **6.2** `core/pose/sync.py` — alinhar pose com `RhythmGrid`
- [ ] **6.3** `core/pose/export.py` — renderizar skeleton OpenPose-style
- [ ] **6.4** Keypoints de pose → SAM points (feed do 4.1)
- [ ] **6.5** LSTM leve treinado em AIST++ (futuro)
- [ ] **Deps** `requirements-pose.txt` com `mediapipe>=0.10` (opcional, fora do requirements principal)

---

## Decisões tomadas

- `mido` sim, `pretty_midi` pular (redundante).
- `mediapipe` só no Sprint 5, em requirements separado.
- MIDI coexiste com áudio: sobrescreve apenas ritmo; features espectrais continuam ativas para timbre.
- Refactor de `VisualObject` entra no Sprint 1 junto do 1.3 (evita retrabalho em 4.x/5.x).
- Cache de rhythm reutiliza o padrão de `file_hash` de `_separate_stems`.

## Riscos / questões em aberto

- ~~Testes: repo sem pytest. Introduzir junto ao 1.1.~~ ✅ `.venv/` criado com pytest + requirements completos.
- Fractais (5.5): manter CPU + upscale; evitar GPU no v1.
- MediaPipe (6.x): wheels frágeis em máquinas sem CUDA — manter isolado.

## Ambiente

- `.venv/` na raiz (gitignored). Python 3.10.
- Rodar testes: `.venv/Scripts/python.exe -m pytest tests/`
- Rodar gerador: `.venv/Scripts/python.exe animation_generator.py --file <audio>`

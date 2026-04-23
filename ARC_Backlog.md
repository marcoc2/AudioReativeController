# ARC — Backlog

> Ideias geradas no brainstorm que **não existem ainda** no repositório.
> Ordenadas por dependência: épicos de cima desbloqueiam os de baixo.

---

## 🧱 ÉPICO 1 — RhythmGrid (Fundação)
> Tudo depende disso. Implementar primeiro.

| # | Tarefa | Detalhes | Deps |
|---|--------|----------|------|
| 1.1 | `core/rhythm/grid.py` | Classe `RhythmGrid`: BPM, fórmula de compasso, duração em segundos e frames, subdivisões de whole até fusa (64th), métodos `phase()`, `is_beat()`, `is_downbeat()` | — |
| 1.2 | `core/rhythm/analyzer.py` | Detecção automática via `librosa.beat.beat_track()`. Retorna `RhythmGrid` populado com beats e downbeats | 1.1 |
| 1.3 | Refatorar loop do `animation_generator.py` | Substituir `time_sec` absoluto por `grid.phase()` e `grid.is_downbeat()`. Animação passa a "saber" onde está no compasso | 1.2 |
| 1.4 | Argumento `--bars` no CLI | Renderizar N compassos completos em vez de N segundos | 1.3 |
| 1.5 | Argumento `--subdivision` no CLI | Define resolução mínima do grid (`quarter`, `eighth`, `16th`…) | 1.3 |

---

## 🎹 ÉPICO 2 — Input MIDI
> Para músicas próprias do Reaper: BPM e velocity exatos, sem estimativa.

| # | Tarefa | Detalhes | Deps |
|---|--------|----------|------|
| 2.1 | `core/rhythm/midi_reader.py` | Lê MIDI via `mido`. Extrai BPM exato, timestamps e velocity de cada nota. Mapeia kick (C2) como downbeat automaticamente | 1.1 |
| 2.2 | Argumento `--midi` no CLI | Quando passado, substitui o analyzer automático pelo MIDI reader | 2.1 |
| 2.3 | Mapeamento nota→objeto | Cada track/canal MIDI pode controlar um objeto visual independente. Ex: canal 10 (bateria) → círculo esquerdo, canal 1 (melodia) → círculo direito | 2.1, 1.3 |
| 2.4 | Velocity → intensidade visual | `velocity 127` = raio máximo, `velocity 60` = raio proporcional. Substituir energia estimada por áudio quando MIDI disponível | 2.3 |

---

## 🔬 ÉPICO 3 — Novas Features de Áudio
> Ampliar o `feature_extractor.py` além de bass e stems.

| # | Tarefa | Detalhes | Deps |
|---|--------|----------|------|
| 3.1 | Spectral Centroid | "Brilho" do som. Já disponível no librosa (`librosa.feature.spectral_centroid`) | — |
| 3.2 | Chroma Features | Detecta qual nota/acorde está tocando. Base para mapear harmonia em cor | — |
| 3.3 | Spectral Flux | Velocidade de mudança do espectro. Detecta transições e cortes | — |
| 3.4 | Onset Detection | Timestamp exato de cada ataque de nota. Mais preciso que beat tracking para sons não-periódicos | — |
| 3.5 | Energia por sub-banda | Sub-bass, low-mid, high-mid, presence separados (além do bass atual) | — |
| 3.6 | Segment Detection | `librosa.segment.agglomerative()` para detectar intro/verso/refrão automaticamente e disparar mudanças de cena | 1.3 |

---

## 🎨 ÉPICO 4 — Novos Outputs de Semente (Seeds)
> Novos formatos exportados em paralelo com o MP4 principal.

| # | Tarefa | Detalhes | Deps |
|---|--------|----------|------|
| 4.1 | SAM Point Generation | `core/output/sam_export.py`. Exporta JSON por frame com `{frame, t, bar_phase, objects: [{id, x, y, r}]}`. Permite usar SAM no ComfyUI para máscaras precisas | 1.1 |
| 4.2 | Alpha Depth Mode | `core/output/depth_map.py`. Renderiza objetos como gradiente de brilho (centro branco = Z=0, bordas pretas = Z=max). ControlNet Depth interpreta como objeto saindo da tela | — |
| 4.3 | Optical Flow Map | `core/output/flow_map.py`. Gera mapa de cores por direção de movimento de pixel. Fundamental para consistência temporal no AnimateDiff | 1.3 |
| 4.4 | Argumento `--outputs` no CLI | `--outputs video,depth,sam,flow`. Renderiza múltiplos formatos em paralelo no mesmo passo | 4.1, 4.2, 4.3 |

---

## 🔷 ÉPICO 5 — Novos Primitivos Visuais
> Expandir além dos círculos atuais.

| # | Tarefa | Detalhes | Deps |
|---|--------|----------|------|
| 5.1 | Polígonos Mutantes | Número de vértices controlado por frequência: bass=triângulo, mid=quadrado, agudo=círculo (N vértices) | 1.3 |
| 5.2 | Vector Fields | Objetos navegam em campo de força perturbado por picos de frequência. Movimento "líquido" em vez de zigzag mecânico | 1.3 |
| 5.3 | Multi-objeto por stem | Cada stem (bass, vocals, drums, outros) vira um objeto independente com física e forma próprias | 1.3, 5.1 |
| 5.4 | L-Systems Rítmicos | Batidas disparam iterações de crescimento fractal (galhos, ramificações). Ideal para ControlNet Canny/Depth | 1.1 |
| 5.5 | Fractais de Áudio | Controlar zoom/iteração de Julia ou Mandelbrot via energia de stems. Gera texturas densas para orientar a IA | — |

---

## 🕺 ÉPICO 6 — Pose Reativa (Pesquisa)
> Mais exploratório. Não bloqueia nada acima.

| # | Tarefa | Detalhes | Deps |
|---|--------|----------|------|
| 6.1 | Pipeline de extração de pose | Usar MediaPipe para extrair keypoints de vídeo de dança gravado pelo usuário | — |
| 6.2 | Sincronizar pose com áudio | Alinhar timestamps do vídeo de dança com RhythmGrid da música | 1.2, 6.1 |
| 6.3 | Exportar pose como seed | Renderizar skeleton no estilo OpenPose (pontos + linhas). ControlNet OpenPose interpreta diretamente | 6.2 |
| 6.4 | Pontos de pose como SAM points | Os keypoints do skeleton também alimentam o SAM export. Pose + segmentação na mesma pipeline | 6.3, 4.1 |
| 6.5 | Modelo LSTM leve (futuro) | Treinar no dataset AIST++ para gerar pose sintética a partir de audio features, sem precisar gravar | 6.2 |

---

## 📦 Dependências Novas Necessárias

```
# a adicionar no requirements.txt
mido>=1.3          # MIDI reader (épico 2)
pretty_midi>=0.2   # alternativa mais alto nível ao mido
mediapipe>=0.10    # extração de pose (épico 6)
```

> `librosa` já está no requirements.txt — os épicos 1 e 3 podem começar sem instalar nada novo.

---

## Ordem Sugerida de Implementação

```
1.1 → 1.2 → 1.3    (RhythmGrid funcional)
      ↓
   3.1–3.5          (novas features de áudio, paralelo)
   2.1 → 2.2–2.4    (MIDI, paralelo)
      ↓
   4.1–4.4          (novos outputs)
   5.1–5.3          (novos primitivos)
      ↓
   3.6              (segment detection, usa grid maduro)
   6.1–6.4          (pose, quando o resto estiver estável)
```

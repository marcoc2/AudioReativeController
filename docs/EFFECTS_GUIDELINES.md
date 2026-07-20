# Guidelines: adicionando efeitos visuais ao ARC

Público-alvo: agentes LLM (e humanos) estendendo o sistema sem quebrar a
arquitetura. Leia inteiro antes de escrever código. Em caso de conflito
entre este documento e o código, o código vence — atualize este arquivo.

## 1. O mapa: quatro pontos de extensão

Todo efeito novo entra por exatamente UM destes seams. Escolher o seam
certo é a decisão mais importante.

| Seam | O que é | Quando usar | Modelos de referência |
|------|---------|-------------|----------------------|
| **Layer source** | Gerador de imagem: produz um frame RGB por chamada | O efeito desenha conteúdo próprio (fractais, partículas, geometria) | `core/cells.py`, `core/fractal.py`, `core/mandelbulb.py`, `core/mandelbox.py` + wrappers em `core/video/layers.py` |
| **Post-op** | Transforma o composto acumulado (pixels) | O efeito modifica a imagem existente (aberração, feedback/túnel, hue shift, blur) | `RgbSplit` em `core/video/layers.py` |
| **Transport action / GravityWarp** | Muda o playback dos clipes (tempo) | O efeito é temporal: direção, velocidade, salto de clipe | `ACTIONS` e `GravityWarp` em `core/video/composer.py` |
| **Legado (congelado)** | `particle_generator*.py`, `animation_generator.py`, `core/particles*.py`, `core/visual.py` | **Nunca estenda.** São modos standalone antigos, despachados pela GUI. Efeitos deles migram virando layer source | — |

Fluxo de dados (não altere): `clip_generator.py` monta grid+notes+
features → `build_compositor()` (único ponto de registro) → loop chama
`stack.frame_at(t)` → pipe rawvideo pro ffmpeg. A GUI não precisa de
mudança para efeitos novos: cenas YAML aparecem de graça.

## 2. Contratos obrigatórios (quebrar = rejeitar o patch)

1. **`frame_at(t) -> np.uint8 (H, W, 3)`** exatamente no tamanho pedido;
   post-ops implementam `process(frame, t) -> frame` (mesmo shape/dtype).
2. **t é monotônico**, uma chamada por frame de saída. Estado interno é
   permitido (playheads, fases, populações), mas derive movimento de `t`
   e `1/fps` — **nunca de wall-clock** (`time.time()` é proibido).
3. **Determinismo**: mesmo áudio+cena+seed ⇒ mesmo vídeo. Aleatoriedade
   só via `seed` no spec (`np.random.default_rng(seed)` / `random.Random`).
4. **Áudio entra por duas portas, apenas**:
   - contínuo: `features_at(t) -> dict` (chroma, flux, centroid,
     subbands, stems…). Suavize com low-pass (`s += (x-s)*(1-exp(-dt/tau))`)
     antes de controlar câmera/movimento — valores crus por frame causam
     jitter (lição aprendida no mandelbulb).
   - discreto: hits via `_layer_hits(spec, notes, onset_loader)` — aceita
     `notes:` (MIDI) e `audio:` (onsets de stem) uniformemente. Envelopes
     com `EnvelopeOpacity`. Nunca parseie MIDI/áudio por conta própria.
5. **Tempo musical** via `grid` (bar_duration, downbeats) — passe pelo
   parâmetro `grid` de `build_compositor` (modelo: `MandelboxLayer.loop_bars`).
6. **Loop-safety**: se o efeito promete loop perfeito, toda função visual
   deve ser periódica no período (posição de mundo, não acumuladores).
   Prove com teste de costura (`test_mandelbox_infinite_zoom_loops_seamlessly`).
7. **Orçamentos**: CPU ≤ ~50ms/frame a 480²; GPU via moderngl standalone
   (modelo: `mandelbulb.py`), supersample opcional. RAM de post-ops com
   histórico (feedback/ecos): declare e limite o buffer (N frames × H×W×3;
   cap N no spec). Nada de vazamento por frame — RSS deve ficar plano.
8. **YAML**: chaves snake_case com defaults seguros; **cena antiga nunca
   muda de comportamento** (default = comportamento atual). Documente o
   bloco no docstring de `layers.py` e/ou `composer.py`.
9. **Registro**: um único branch novo em `build_compositor` (`source:
   <nome>`); post-op não pode ser a camada base (o `Compositor` valida).
10. **Encadeamento**: a ordem da lista `layers:` é a ordem de composição
    (de baixo pra cima); post-ops agem sobre tudo que está abaixo deles
    na lista. Efeitos devem ser projetados para empilhar — identidade
    quando intensidade/envelope = 0.

## 3. Testes mínimos por efeito (padrões em `tests/test_clip_video.py`)

- shape/dtype corretos; **identidade quando desligado** (envelope 0 /
  opacity 0 / amount 0);
- resposta a controls (frame muda quando o áudio muda);
- consumo de trigger (hit produz o efeito no instante certo);
- determinismo com seed; costura de loop quando aplicável;
- GPU: `pytest.skip` gracioso quando não há contexto GL.
Rode `pytest tests/ -q` (tudo verde) e um render curto real de validação
(6 compassos, 480²) com estatística de pixel provando o efeito (ex.:
brilho no hit vs fora dele — ver histórico de commits por exemplos).

## 4. Checklist de entrega

- [ ] Seam certo escolhido (tabela §1) e justificado no commit
- [ ] Contratos §2 respeitados (releia os 10)
- [ ] Testes §3 + suite completa verde
- [ ] Cena de exemplo em `examples/` (+ .md com CLI e passos de GUI) se
      o efeito for "flagship"; senão, documente no docstring do YAML
- [ ] `examples/README.md` atualizado se criou exemplo
- [ ] Render curto de validação inspecionado (frame extraído)

## 5. Mapeamento das sugestões propostas (Gemini) → seams

Estas cinco ideias são compatíveis com a arquitetura; caminho sancionado:

1. **Mandalas aninhadas / orbitadores em cascata** → *layer source* novo
   (`source: orbiters`): órbitas principais por `bar_phase` (via `grid`),
   satélites por `centroid`/`flux`. NÃO estenda `animation_generator.py`
   (legado) — porte a ideia como camada.
2. **Emissores de partículas nos vértices do polígono** → *layer source*
   composto: o port de partículas-como-camada (pendente) deve expor
   pontos de emissão; vértices vêm da própria camada (geometria interna),
   jatos disparados por hits via `_layer_hits`. Não acople duas camadas
   por estado global — se precisar compartilhar geometria, é UMA camada
   com dois subsistemas.
3. **Feedback loop / túnel (zoom+rotação acumulados)** → *post-op com
   estado* (guarda `self._prev`): compõe o frame anterior re-escalado/
   rotacionado sob o atual. Zoom ← envelope de kick; rotação ← `centroid`
   suavizado. RAM: 1 frame retido (declare). Atenção ao contrato 10:
   identidade quando `strength: 0`.
4. **Ecos de rastro dinâmicos (doppler de tamanho + hue shift)** →
   *post-op com ring buffer* (N frames, cap via spec `depth`, default
   pequeno; RAM = N×frame). Hue shift por `flux` do frame atual.
5. **Aberração cromática reativa a transientes** → **já implementada**
   como referência: `source: rgb_split` (`RgbSplit` em
   `core/video/layers.py`) — envelope de trigger (MIDI ou onset de
   áudio) desloca R/B; identidade em repouso. Use-a como gabarito de
   post-op.

```yaml
# exemplo de encadeamento completo (ordem importa):
video:
  layers:
    - source: clips                    # base
    - source: cells                    # gerador por cima (blend screen)
      blend: screen
    - source: rgb_split                # post-op sobre tudo acima
      amount: 8
      trigger: {notes: [38, 40], envelope: 0.12}
```

## 6. Status do legado (`particle_generator*.py`)

Verificado em 2026-07-19: são usados por (a) despacho de modos da GUI
(`_MODE_SCRIPTS` no `arc_studio.py`), (b) hotkey [R] do
`visualizer_debug.py` (animation_generator), (c) testes de comando.
São o único caminho até `core/particles*.py` hoje. Política: mantê-los
funcionando, **não adicionar features**; a migração sancionada é
embrulhar os `ParticleSystem*` como layer sources e, então, aposentar os
scripts standalone.

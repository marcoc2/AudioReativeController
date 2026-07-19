# Enxame — Clipes com passagem de bastão kick→snare (`enxame_clips_handover.yaml`)

Miniclipes pré-renderizados trocando por eventos de bateria: o **bumbo
(MIDI 36)** comanda `next_clip` na introdução; quando a **caixa gravada**
entra (~43.1s, onsets detectados no stem com `exclude` matando o bleed
do bumbo), ela assume sozinha até o fim (`until: snare`). Ordem dos
clipes em shuffle (re-embaralha por ciclo, sem repetição imediata).

## CLI

```powershell
.venv\Scripts\python.exe clip_generator.py `
  --file "C:\Audio\recordings\projeto2024\ep\stems\enxame\mix_enxame_ep_15_07_2026.mp3" `
  --midi "C:\Audio\recordings\projeto2024\ep\stems\enxame\enxame_drum_no_snare.mid" `
  --clips <SUA_PASTA_DE_CLIPES> `
  --scene examples/enxame_clips_handover.yaml `
  --bars 0 --cache-size 101 --resolution 480x480 `
  --output render_output/enxame_clips.mp4
```

`--bars 0` = música inteira; `--seed N` fixa o sorteio do shuffle.

## Na GUI (ARC Studio)

1. `python arc_studio.py --project projects/enxame.yaml` já abre tudo
   configurado (é este cenário). Ou monte na mão:
2. **Audio** = mix, **MIDI** = `enxame_drum_no_snare.mid`, **Scene** =
   `examples/enxame_clips_handover.yaml`; painel STEMS com o stem da
   caixa (aparece como waveform na timeline).
3. **Mode: clips** → aparecem **Clips Folder** (sua pasta), Order,
   Full song, Cache. **Load Project**.
4. Na timeline: lane **TRIGGERS** (ticks laranja = kick, roxo = snare),
   linha amarela **"snare assume"** em ~43s, lane **SAIDA** com o
   arranjo resolvido — e o **CLIP DECK** (Build Deck) permite arrastar
   thumbnails pra pinar compassos (📌, salvo no projeto).
5. **Preview** ouve/vê um trecho; **Render Full** gera o vídeo.

O editor **SCENE — CLIP TRIGGERS** mostra/edita os dois triggers desta
cena; "Save Scene" grava o YAML e atualiza a timeline na hora.

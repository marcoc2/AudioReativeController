# Backlog de bugs e dívidas

Problemas encontrados no uso que ainda não foram resolvidos. Cada item diz como
aparece, por que acontece e a correção proposta.

## 1. [ ] `snap: auto` engole acordes que trocam no meio do compasso

**Como aparece.** Em me_diz, o túnel da `me_diz_acorde` ficou cinza no Sol de 24.3, que
deveria colorir por ser um acorde maior. O Sol simplesmente sumiu.

**Por quê.** Nessa música a harmonia troca no tempo 3 dos compassos do DAW. O
`snap: auto` (o padrão quando há MIDI) puxa cada troca para a linha de compasso mais
próxima. Uma troca exatamente no meio do compasso fica a igual distância das duas
linhas: o Sol foi para frente, o Si7 seguinte foi para trás, e os dois caíram na mesma
linha. O Sol ficou com duração zero.

```
           24               25               26
           |---+---+---+----|---+---+---+----|
soa                |G...............|B7.......
snap: auto Em...............|B7............     ← o G sumiu
snap: beat Em......|G...............|B7.......  ← certo
```

**Contorno hoje.** `snap: beat` na cena. A `me_diz_acorde` já usa; a `me_diz_capa`
ainda usa `auto` na matiz do túnel e perde esses Sóis.

**Correção proposta.** Quando dois acordes caírem na mesma linha de compasso, o
`auto` usa o tempo em vez do compasso, para não engolir nenhum. Músicas que trocam no
tempo 1 continuam iguais.

## 2. [ ] Efeitos presos à régua do compasso, e não à música

**Como aparece.** Em me_diz, os cortes de clipe (`bars_per_clip: 2`) caem no meio de
cada acorde, porque a harmonia troca no tempo 3 e os cortes caem no 1.

**Por quê.** Os efeitos disparados por nota (kick, caixa, baixo, `wake` com
`hold: note`) já caem no instante exato da nota, fora do grid. Só estas partes ficaram
presas ao compasso:
- os cortes de clipe (`bars_per_clip`);
- os acordes e seções da transcrição (`snap`);
- os knobs `bar:` e `count` por compasso;
- o `wake` com `hold: bar`.

**Pergunta em aberto.** Em me_diz, a música é frasada a partir do tempo 3, ou o
compasso 1 do projeto do DAW não coincide com o "1" musical?

**Correções propostas** (pequenas, todas sobre a mesma base de grid):
- **Cortes que seguem a música.** `cut_on: chords` (cortar na troca de acorde) ou
  `cut_offset: 2 beats`. Hoje já é possível com um trigger `score:` usando
  `next_clip`, mas é trabalhoso.
- **Deslocamento em qualquer trigger.** `offset: -1/16` antecipa o efeito uma
  semicolcheia, como uma antecipação na música; `offset: 1/8` atrasa. O render é
  offline, então adiantar não custa nada.
- **Padrões rítmicos como gatilho**, sem nota MIDI por trás. Por exemplo,
  `pattern: "x..x..x.x......."` em semicolcheias faz uma clave 3-3-2.

# Shaders do Shadertoy

Shaders de [shadertoy.com](https://www.shadertoy.com) para rodar no ARC como camada
(`source: shadertoy`, adaptador em `core/shadertoy.py`).

## Um arquivo por shader (ou uma pasta, se tiver buffers)

- Só a aba **Image**: `<nome>.glsl`.
- Com **Buffer A..D** e/ou **Common**: uma pasta `<nome>/` com `image.glsl`,
  `buffer_a.glsl` … `buffer_d.glsl` e `common.glsl` (cada aba num arquivo, com o
  nome da aba). Exemplo: `sombras_2d/`.

O código fica **como está no Shadertoy** (não edite o original; se precisar mudar,
salve uma cópia `<nome>_arc` e anote o que mudou). No topo de cada arquivo, o
cabeçalho em linhas `// chave: valor`:

```glsl
// title: Nome do shader
// url: https://www.shadertoy.com/view/XXXXXX
// author: usuario_no_shadertoy
// license: CC BY-NC-SA 3.0
// iChannel0: frame
// notes: o que ele faz, o que foi mudado (se for cópia _arc)

void mainImage(out vec4 fragColor, in vec2 fragCoord) { ... }
```

- **Canais**: `// iChannel0: ...` a `// iChannel3: ...` dizem o que cada canal daquela aba
  lê (o que no Shadertoy fica nos quadradinhos embaixo do código):
  - `buffer_a` … `buffer_d`: um buffer. Como no Shadertoy, os buffers rodam antes da Image,
    em ordem, em textura float (valores negativos e acima de 1 ficam); um buffer que lê a
    si mesmo recebe o próprio quadro anterior (feedback).
  - `frame`: o quadro de baixo (o clipe). A camada vira pós-processamento.
  - `previous`: a última saída da própria Image.
  - `sound`: a música como a entrada de som do Shadertoy (espectro na linha de cima,
    forma de onda na de baixo), calculada dos recursos de áudio do quadro.
  - o caminho de uma imagem (textura fixa).
  Na cena, `channel0..3` sobrescrevem os canais da aba Image.
- Não suportados: cube maps, teclado. `iMouse` fica parado em (0, 0); se o shader
  usa o mouse, dá para ligar a posição na música numa cópia `_arc`.

## Licença: leia antes de publicar

Todo shader do Shadertoy é, **por padrão, CC BY-NC-SA 3.0**, a menos que o autor diga
outra coisa no código:

| | |
|---|---|
| **BY** | crédito ao autor (nome + link) na descrição do vídeo |
| **NC** | **sem uso comercial**: vídeo monetizado ou trabalho pago já esbarra |
| **SA** | o que derivar herda a mesma licença |

Muitos autores liberam em MIT, CC0 ou domínio público no próprio código: confira e
anote em `license:`. Sem licença clara, trate como CC BY-NC-SA 3.0 (estudo e
referência; não vai para vídeo publicado sem autorização do autor).

## Na cena

```yaml
- source: shadertoy
  file: shaders/shadertoy/nome.glsl
  channel0: frame                  # o clipe entra no shader
  speed: 1.0                       # o relógio do shader (iTime)
  rush: {hits: {track: kick}, amount: 3.0, envelope: 0.3}   # o relógio dispara no kick
  uniforms:                        # botões próprios (float), declarados se o shader não declarar
    u_bass: subbands.bass          # um recurso do áudio
    u_kick: {hits: {track: kick}, envelope: 0.25}           # pulso 1 -> 0 a cada batida
    u_count: {count: {track: kick}, per: bar}               # batidas já tocadas no compasso (0 no início)
    u_note: {pitch: {track: bass}, low: 48, high: 53, glide: 0.08}   # altura da última nota, 0..1
    u_spin: {rate: 4.0, base: 0.3, from: {hits: {track: bass}, envelope: 0.2}}
                                   # um ângulo que não para: base + rate * (outro botão) por segundo
    u_bar: {bar: {step: 1, morph: 0.35}}                    # sobe 1 a cada compasso, derretendo
    u_since: {since: {track: snare}}                        # segundos desde a última caixa
    u_hue: {harmony: {score: input/musica/sheetsage}, glide: 1.2}   # matiz da tônica do acorde
    u_third: {chord: {score: input/musica/sheetsage}, part: tone, tone: 1, glide: 0.5}
                                   # a terça do acorde, 0..11 (C = 0)
    u_mode: {chord: {score: input/musica/sheetsage}, part: mode}    # 1 maior, 0 menor, 0.5 sus
    u_cut: {cut: until}            # segundos até o próximo corte de clipe
```

**`chord:`** lê o acorde que soa agora na transcrição do SheetSage. Nos trechos sem
acorde, ele mantém o último. Cada `part` dá um número:

| `part` | o que vale |
|---|---|
| `root` | a tônica, 0..11 (C = 0) |
| `tone` | a nota `tone` do acorde (0 tônica, 1 terça, 2 quinta, 3 sétima…); num acorde menor que isso, dá a volta |
| `size` | quantas notas (3 tríade, 4 tétrade…) |
| `mode` | 1 com terça maior, 0 com terça menor, 0.5 sem terça (sus, power chord) |
| `tension` | 0 numa tríade simples, até 1 (diminuto/aumentado, sus, notas acrescentadas) |

`glide` desliza até o valor novo em segundos. `root` e `tone` deslizam pelo caminho
mais curto da roda de 12. Com `glide: 0` (o padrão), o valor salta.

**`cut:`** segue os cortes de clipe. O compositor sabe de antemão onde eles caem: nos
compassos de `bars_per_clip`, nos compassos fixados em `overrides` e nos triggers com
`next_clip` ou `random_clip`. Por isso dá para começar uma mudança antes do corte.

| `cut:` | o que vale |
|---|---|
| `count` | cortes até agora (0 no primeiro plano); `morph: s` sobe suave |
| `since` | segundos desde o último corte (1000 antes do primeiro) |
| `until` | segundos até o próximo (1000 se não houver) |
| `phase` | 0 → 1 ao longo do plano |

`cut:` precisa de uma cena com `- source: clips`.

Um shader só responde a um `uniforms:` se o usar: para ligar um botão num shader
baixado, salve a cópia `_arc` e troque uma constante pelo uniform.

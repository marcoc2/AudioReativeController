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
  - o caminho de uma imagem (textura fixa).
  Na cena, `channel0..3` sobrescrevem os canais da aba Image.
- Não suportados: cube maps, som, teclado. `iMouse` fica parado em (0, 0); se o shader
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
```

Um shader só responde a um `uniforms:` se o usar: para ligar um botão num shader
baixado, salve a cópia `_arc` e troque uma constante pelo uniform.

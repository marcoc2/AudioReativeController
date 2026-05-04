# ARC: New Backlog & Insights 🚀

Este documento serve para registrar ideias, melhorias arquiteturais e insights técnicos surgidos durante o desenvolvimento e onboarding de novos membros.

## 1. Cinemática de Features (Derivadas de Áudio)

### Descrição
Atualmente, o `AudioFeatureExtractor` fornece dados "instantanecistas" (energia no tempo T). Para criar animações mais orgânicas e impactantes, precisamos entender a dinâmica de mudança desses dados.

### User Story
**Como** um desenvolvedor de motion graphics,
**Eu quero** acessar a velocidade (1ª derivada) e a aceleração (2ª derivada) de cada feature de áudio (bass, vocal, etc.),
**Para que** eu possa implementar efeitos de "impacto" (baseados em aceleração) e "rastros/motion blur" (baseados em velocidade).

### Notas Técnicas
- **Velocidade:** `V = Features(T) - Features(T-1)`.
- **Aceleração:** `A = V(T) - V(T-1)`.
- **Implementação:** Necessário expandir o `AudioFeatureExtractor` para manter um histórico de pelo menos dois frames anteriores (`prev_features` e `prev_prev_features`) ou armazenar o vetor de velocidade do frame anterior.
- **Insight:** O `Spectral Flux` já atua como uma velocidade global do espectro, mas precisamos disso de forma granular para cada banda e stem.

---

## 2. Refatoração da "God Class" (AudioFeatureExtractor)

### Descrição
A classe `AudioFeatureExtractor` possui muitos atributos (28+) e responsabilidades misturadas (DSP, Gerenciamento de Cache de IA, Física de Suavização).

### User Story
**Como** um mantenedor do projeto,
**Eu quero** que a lógica de extração de áudio seja separada da lógica de física/suavização,
**Para que** o código seja mais fácil de testar e expandir.

### Notas Técnicas
- Criar classes de dados (DataClasses) para `AudioMetadata`.
- Isolar o `SmoothingEngine` em um componente separado.

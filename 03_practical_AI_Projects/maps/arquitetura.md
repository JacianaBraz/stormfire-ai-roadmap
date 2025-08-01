# 🧠 Arquitetura NPL (Natural Language Processing)

## 📚 Tipos de Arquiteturas

### 🔄 RNN (Recurrent Neural Networks)
- **Propósito**: Projetada para lidar com dados sequenciais
- **Característica**: Processa informações em sequência temporal

### 🧠 LSTM (Long Short-Term Memory)
- **Função**: Lembrar informações por mais tempo
- **Componentes**:
  - `cell state` (memória de longo prazo)
  - `hidden state`

### ⚡ GRU (Gated Recurrent Unit)
- **Função**: Mesma função do LSTM, usando menos recursos
- **Componentes**:
  - `hidden state`

---

## 🤖 Transformers

> Criada para lidar com dados sequenciais, como texto, mas sem depender da sequência. Usam o **Self-Attention** para conectar tudo com tudo.

### 📥 Encoder
**Função**: Entender o input, seja uma frase, uma imagem ou o que for. Ele não gera resposta. Ele observa, absorve, interpreta.

#### 🔧 O que ele faz:
1. Recebe a sequência de entrada (ex: "O céu está azul")
2. Aplica `embeddings` + `positional encoding` → transforma palavras em vetores com posição
3. Passa isso por vários blocos de atenção (`Self-Attention`) e `Feed-Forward Networks`
4. **Resultado**: Um conjunto de vetores ricos em contexto → chamados de "context vectors"

#### 💡 Exemplo: BERT
**BERT** (Bidirectional Encoder Representations for Transformers)

- **Serve para**: Compreender, classificar, responder perguntas, extrair entidades, etc.
- **Não faz**: Não prevê próxima palavra
- **Treinamento**: Usa Máscara (`Masked Language Modeling`) → ele esconde palavras e tenta adivinhar com base no contexto

---

### 📤 Decoder
**Função**: Pega essa informação codificada e gera a resposta, palavra por palavra, passo a passo. Ele é o output.

#### 🔧 O que ele faz:
1. Começa com um token especial `[START]` (ou o que já foi gerado)
2. Usa `self-attention` para olhar para o que ele já produziu
3. Usa `encoder-decoder attention` para olhar para o que o encoder entendeu
4. Passa por um `feed-forward` e gera o próximo token
5. Repete até chegar no token `[END]`

#### 💡 Exemplo: GPT
- **Arquitetura**: Usa somente o decoder, com uma técnica chamada `masked self-attention` (ele só olha pro passado)
- **Serve para**: Gerar texto, completar sentenças, criar respostas, escrever código, etc.
- **Limitações**:
  - Ele não entende texto inteiro ao mesmo tempo, só prevê palavra por palavra
  - Precisa de muito contexto anterior para manter coerência
  - Mais propenso a "alucinar" (inventar coisa), porque não entende tudo ao mesmo tempo como o BERT

---

### 🔄 Encoder-Decoder
**Conceito**: O encoder lê toda a entrada e gera um "mapa de entendimento". O decoder consulta esse mapa enquanto vai construindo a saída.

#### 💡 Exemplo: T5 (Text-To-Text Transfer Transformer)

##### ✨ Características Principais:

**🎯 Formato único para todas as tarefas**
- Um único modelo pode fazer MUITA coisa
- Isso facilita fine-tuning

**🎭 Pré-treinamento com "span corruption"**
- Diferente do BERT que mascara só UMA palavra, o T5 mascara TRECHOS inteiros (chamado span corruption)
- Isso ensina o modelo a reconstruir ideias complexas, não só palavras

**🚀 Flexibilidade absurda**
- Pode usar o mesmo modelo para tradução, QA, inferência, resumo, classificação, etc.

**📝 Instrução explícita embutida no input**
- Ele precisa que você diga o que quer dele
- Isso é a base de tudo que virou prompt engineering depois

**🌟 Base para**:
- `FLAN-T5` → versão refinada, com fine-tuning em múltiplas tarefas usando instruções
- `Alpaca`, `Vicuna`, `LLaMA-Instruction`

**🎓 Treinamento**: Text-to-text com instruções

---

#### 💡 Exemplo: BART (Bidirectional and Auto-Regressive Transformer)

##### 🔧 Como funciona?

**🎯 Pré-treinamento com reconstrução de texto corrompido**
- Ele aprende a consertar frases bagunçadas

**🛠️ Técnicas como**:
- Adicionar ruído (corrupção)
- Embaralhar sentenças
- Apagar palavras aleatórias

**📥 Encoder (BERT-like)**
- Processa a entrada com compreensão bidirecional
- Ele "lê" o texto todo antes de tomar decisões

**📤 Decoder (GPT-like)**
- Gera a saída palavra por palavra, prevendo o próximo token com base nos anteriores

##### ⭐ O que o torna especial?

**🎯 Especialista em tarefas de reconstrução de texto**
- Resumo, correção gramatical, geração condicional, reescrita

**🧠 Compreende o contexto de forma rica (como BERT)**
- Não depende só do que veio antes (como GPT), mas também do que veio depois
- Isso é ouro para entender o meio da frase

**✍️ Gera texto com fluidez (como GPT)**
- O decoder autoregressivo permite respostas coerentes, longas, naturais

---

#### 💡 Exemplo: MarianMT (Marian Machine Translation)

##### 🔧 Como ele funciona?

**🏗️ Arquitetura baseada no Transformer (Encoder–Decoder)**
- Pré-treinado em grandes corpora paralelos, como:
  - **OPUS**: banco de dados de traduções humanas
- Cada modelo é direcionado para um par de línguas (ex: `pt-en`, `en-de`, etc.)

**🧠 Processo**:
- Ele usa o encoder para entender o idioma de origem
- E o decoder para construir a tradução, uma palavra por vez, no idioma-alvo

##### ⭐ O que torna especial?

**🎯 Treinamento altamente especializado**
- Focado só em tradução, sem ruídos de outras tarefas

**⚡ Modelos pequenos e eficientes**
- Leves o bastante para rodar em CPUs ou até em mobile, mas ainda potentes

**🌍 Multilíngue com escalabilidade**
- Pode lidar com centenas de pares linguísticos

**🤗 Compatível com HuggingFace**
- Prontinho para você testar com apenas 3 linhas de código

---

## 🔧 Componentes

### 📝 Input

#### 🔤 Tokenização
> Processo de quebrar um input em partes menores chamadas tokens

**Tipos de Tokenização**:

- **📝 Word-level**: Cada palavra é um token
- **🔤 Character-level**: Cada letra é um token  
- **🧩 Subword**: Cada montante de letras é um token
  - **BPE** (Byte Pair Encoding): Não respeita espaços, aprende onde o espaço deveria estar a partir do padrão das subpalavras
  - **SentencePiece**: Respeita espaços

---

### 🎯 Embeddings
> Transformam palavras (tokens) em vetores

#### 📊 Modelos

**🔒 Estáticos** (cada palavra tem o mesmo vetor, independente do contexto):
- `Word2Vec`
- `GloVe`

**🔄 Contextual**:
- `ELMo`

---

### 💾 Estados Internos
- `hidden state`
- `cell state` (memória a longo prazo)

---

## ⚙️ Componentes dos Transformers

### 📍 Positional Encoding
> Saber a posição das palavras numa frase

### 👁️ Self-Attention

#### 🔄 Transformação
Cada palavra vira:
- **🔍 Query** (o que quer saber)
- **🔑 Key** (o que oferece)  
- **💎 Value** (o que carrega)

#### 📊 Scoring (Pontuação)
A palavra pega sua query e compara com a key de outras, isso gera um peso. Quanto maior o peso, maior a relevância da palavra.

#### ⚖️ Combinação ponderada (Atenção)
Esse peso é multiplicado ao value, dando mais importância para as mais relevantes. Resultado: um vetor contextualizado.

### 🎭 Multi-Head Attention
> Analisa vários sentidos da frase ao mesmo tempo com múltiplas "cabeças". 
> 
> **PS**: Todos esses sentidos são "cabeças" que passam por um filtro para definir o real.

### 🔧 Feed-Forward Network
> Refina o resultado da atenção, prepara para os próximos blocos.

### 🔄 Residual Coding
> Mantém o que foi aprendido antes, criando atalhos diretos para evitar "esquecimento".

---

## 📊 Visualização

### 📈 PCA (Principal Component Analysis)
- **✅ Utilidade**: Reduz as dimensões dos vetores mantendo o máximo de variância. Ideal para compreensão rápida e aceleração de visualização
- **❌ Limitações**: Não lida bem com padrões não lineares, não preserva estrutura local

### 🎯 t-SNE (t-distributed Stochastic Neighbor Embedding)
- **✅ Utilidade**: Focado em manter amizades locais (clusters próximos). Ótimo para analisar relações semânticas entre palavras
- **❌ Limitações**: Não preserva estrutura global, resultados instáveis, lento com grandes volumes. Serve para visualização, não para input de modelo

### 🗺️ UMAP (Uniform Manifold Approximation and Projection)
- **✅ Utilidade**: Preserva estrutura local e global. Rápido. Usado em pipelines reais de deep learning, NLP, bioinformática
- **🚀 Funções extras**: Pode ser usado para visualização, compressão, clustering e embedding em produção

---

## 📚 Resumo

Este documento apresenta uma visão abrangente das principais arquiteturas de NPL, desde RNNs tradicionais até os modernos Transformers, incluindo seus componentes fundamentais e técnicas de visualização. Cada arquitetura tem suas características específicas e casos de uso ideais, formando o ecossistema atual do processamento de linguagem natural.


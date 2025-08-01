# 🧠 NLP Architecture (Natural Language Processing)

## 📚 Architecture Types

### 🔄 RNN (Recurrent Neural Networks)
- **Purpose**: Designed to handle sequential data
- **Characteristic**: Processes information in temporal sequence

### 🧠 LSTM (Long Short-Term Memory)
- **Function**: Remember information for longer periods
- **Components**:
  - `cell state` (long-term memory)
  - `hidden state`

### ⚡ GRU (Gated Recurrent Unit)
- **Function**: Same function as LSTM, using fewer resources
- **Components**:
  - `hidden state`

---

## 🤖 Transformers

> Created to handle sequential data, like text, but without depending on sequence. Uses **Self-Attention** to connect everything with everything.

### 📥 Encoder
**Function**: Understand the input, whether it's a sentence, an image, or whatever. It doesn't generate responses. It observes, absorbs, interprets.

#### 🔧 What it does:
1. Receives the input sequence (e.g., "The sky is blue")
2. Applies `embeddings` + `positional encoding` → transforms words into vectors with position
3. Passes this through several attention blocks (`Self-Attention`) and `Feed-Forward Networks`
4. **Result**: A set of context-rich vectors → called "context vectors"

#### 💡 Example: BERT
**BERT** (Bidirectional Encoder Representations for Transformers)

- **Used for**: Understanding, classifying, answering questions, extracting entities, etc.
- **Doesn't do**: Doesn't predict next word
- **Training**: Uses Masking (`Masked Language Modeling`) → it hides words and tries to guess based on context

---

### 📤 Decoder
**Function**: Takes this encoded information and generates the response, word by word, step by step. It is the output.

#### 🔧 What it does:
1. Starts with a special token `[START]` (or what has already been generated)
2. Uses `self-attention` to look at what it has already produced
3. Uses `encoder-decoder attention` to look at what the encoder understood
4. Passes through a `feed-forward` and generates the next token
5. Repeats until reaching the `[END]` token

#### 💡 Example: GPT
- **Architecture**: Uses only the decoder, with a technique called `masked self-attention` (it only looks at the past)
- **Used for**: Generating text, completing sentences, creating responses, writing code, etc.
- **Limitations**:
  - It doesn't understand entire text at once, only predicts word by word
  - Needs a lot of previous context to maintain coherence
  - More prone to "hallucinating" (making things up), because it doesn't understand everything at once like BERT

---

### 🔄 Encoder-Decoder
**Concept**: The encoder reads the entire input and generates an "understanding map". The decoder consults this map while building the output.

#### 💡 Example: T5 (Text-To-Text Transfer Transformer)

##### ✨ Main Characteristics:

**🎯 Unified format for all tasks**
- A single model can do MANY things
- This facilitates fine-tuning

**🎭 Pre-training with "span corruption"**
- Unlike BERT which masks only ONE word, T5 masks ENTIRE CHUNKS (called span corruption)
- This teaches the model to reconstruct complex ideas, not just words

**🚀 Incredible flexibility**
- Can use the same model for translation, QA, inference, summarization, classification, etc.

**📝 Explicit instruction embedded in input**
- It needs you to tell it what you want from it
- This is the foundation of everything that became prompt engineering later

**🌟 Foundation for**:
- `FLAN-T5` → refined version, with fine-tuning on multiple tasks using instructions
- `Alpaca`, `Vicuna`, `LLaMA-Instruction`

**🎓 Training**: Text-to-text with instructions

---

#### 💡 Example: BART (Bidirectional and Auto-Regressive Transformer)

##### 🔧 How does it work?

**🎯 Pre-training with corrupted text reconstruction**
- It learns to fix messed up sentences

**🛠️ Techniques like**:
- Adding noise (corruption)
- Shuffling sentences
- Randomly deleting words

**📥 Encoder (BERT-like)**
- Processes input with bidirectional understanding
- It "reads" the entire text before making decisions

**📤 Decoder (GPT-like)**
- Generates output word by word, predicting the next token based on previous ones

##### ⭐ What makes it special?

**🎯 Specialist in text reconstruction tasks**
- Summarization, grammar correction, conditional generation, rewriting

**🧠 Understands context richly (like BERT)**
- Doesn't depend only on what came before (like GPT), but also on what comes after
- This is gold for understanding the middle of a sentence

**✍️ Generates text fluently (like GPT)**
- The autoregressive decoder allows coherent, long, natural responses

---

#### 💡 Example: MarianMT (Marian Machine Translation)

##### 🔧 How does it work?

**🏗️ Architecture based on Transformer (Encoder–Decoder)**
- Pre-trained on large parallel corpora, such as:
  - **OPUS**: database of human translations
- Each model is targeted for a language pair (e.g., `pt-en`, `en-de`, etc.)

**🧠 Process**:
- It uses the encoder to understand the source language
- And the decoder to build the translation, one word at a time, in the target language

##### ⭐ What makes it special?

**🎯 Highly specialized training**
- Focused only on translation, without noise from other tasks

**⚡ Small and efficient models**
- Light enough to run on CPUs or even mobile, but still powerful

**🌍 Multilingual with scalability**
- Can handle hundreds of language pairs

**🤗 Compatible with HuggingFace**
- Ready for you to test with just 3 lines of code

---

## 🔧 Components

### 📝 Input

#### 🔤 Tokenization
> Process of breaking an input into smaller parts called tokens

**Tokenization Types**:

- **📝 Word-level**: Each word is a token
- **🔤 Character-level**: Each letter is a token  
- **🧩 Subword**: Each chunk of letters is a token
  - **BPE** (Byte Pair Encoding): Doesn't respect spaces, learns where space should be from subword patterns
  - **SentencePiece**: Respects spaces

---

### 🎯 Embeddings
> Transform words (tokens) into vectors

#### 📊 Models

**🔒 Static** (each word has the same vector, regardless of context):
- `Word2Vec`
- `GloVe`

**🔄 Contextual**:
- `ELMo`

---

### 💾 Internal States
- `hidden state`
- `cell state` (long-term memory)

---

## ⚙️ Transformer Components

### 📍 Positional Encoding
> Know the position of words in a sentence

### 👁️ Self-Attention

#### 🔄 Transformation
Each word becomes:
- **🔍 Query** (what it wants to know)
- **🔑 Key** (what it offers)  
- **💎 Value** (what it carries)

#### 📊 Scoring
The word takes its query and compares it with the key of others, this generates a weight. The higher the weight, the greater the relevance of the word.

#### ⚖️ Weighted combination (Attention)
This weight is multiplied by the value, giving more importance to the most relevant ones. Result: a contextualized vector.

### 🎭 Multi-Head Attention
> Analyzes multiple meanings of the sentence simultaneously with multiple "heads". 
> 
> **PS**: All these meanings are "heads" that pass through a filter to define the real one.

### 🔧 Feed-Forward Network
> Refines the attention result, prepares for the next blocks.

### 🔄 Residual Coding
> Maintains what was learned before, creating direct shortcuts to avoid "forgetting".

---

## 📊 Visualization

### 📈 PCA (Principal Component Analysis)
- **✅ Utility**: Reduces vector dimensions while maintaining maximum variance. Ideal for quick understanding and visualization acceleration
- **❌ Limitations**: Doesn't handle non-linear patterns well, doesn't preserve local structure

### 🎯 t-SNE (t-distributed Stochastic Neighbor Embedding)
- **✅ Utility**: Focused on maintaining local friendships (close clusters). Great for analyzing semantic relationships between words
- **❌ Limitations**: Doesn't preserve global structure, unstable results, slow with large volumes. Used for visualization, not for model input

### 🗺️ UMAP (Uniform Manifold Approximation and Projection)
- **✅ Utility**: Preserves local and global structure. Fast. Used in real pipelines of deep learning, NLP, bioinformatics
- **🚀 Extra functions**: Can be used for visualization, compression, clustering and embedding in production

---

## 📚 Summary

This document presents a comprehensive view of the main NLP architectures, from traditional RNNs to modern Transformers, including their fundamental components and visualization techniques. Each architecture has its specific characteristics and ideal use cases, forming the current ecosystem of natural language processing.


# Ngày 3-4: Tokenization, Embedding và LLM Parameters

## 🎯 Mục tiêu học tập
Sau buổi học này, các bạn sẽ hiểu:
- Cách máy tính chuyển đổi text thành số (tokenization)
- Embedding là gì và tại sao quan trọng
- Các parameters điều khiển output của LLM
- Cách áp dụng kiến thức vào thực tế

---

## 📚 Phần 1: Tokenization - Làm thế nào máy tính "đọc" text?

### 🤔 Vấn đề cơ bản
**Máy tính chỉ hiểu số, nhưng chúng ta muốn nó hiểu text. Làm sao?**

### 🔍 Ví dụ đơn giản
```
Text: "Hello world!"
👇 Tokenization
Tokens: ["Hello", " world", "!"]
👇 Convert to numbers
Token IDs: [15496, 995, 0]
```

### 📊 Các phương pháp Tokenization

#### 1. **Word-level Tokenization** (Cách đơn giản nhất)
```
"I love programming" → ["I", "love", "programming"]
```
**Vấn đề:** 
- Từ mới không có trong vocabulary → UNK (Unknown)
- Vocabulary rất lớn (hàng triệu từ)

#### 2. **Character-level Tokenization**
```
"Hello" → ["H", "e", "l", "l", "o"]
```
**Ưu điểm:** Vocabulary nhỏ (chỉ ~100 ký tự)
**Nhược điểm:** Sequences rất dài, khó học patterns

#### 3. **Subword Tokenization** (Phương pháp hiện đại)

**Byte-Pair Encoding (BPE)** - Cách GPT làm:
```
Bước 1: Bắt đầu với characters
"programming" → ["p", "r", "o", "g", "r", "a", "m", "m", "i", "n", "g"]

Bước 2: Tìm cặp xuất hiện nhiều nhất
"mm" xuất hiện nhiều → merge thành "mm"
["p", "r", "o", "g", "r", "a", "mm", "i", "n", "g"]

Bước 3: Lặp lại cho đến khi đủ vocabulary
"programming" → ["program", "ming"]
```

### 💡 Tại sao BPE hiệu quả?
- **Từ thông dụng:** Được giữ nguyên ("the", "and", "is")
- **Từ hiếm:** Được chia nhỏ ("antidisestablishmentarianism" → ["anti", "dis", "establish", "ment", "arian", "ism"])
- **Từ mới:** Vẫn có thể tokenize được

---

## 📚 Phần 2: Embedding - Từ số thành ý nghĩa

### 🎨 Embedding là gì?
**Embedding = Cách biểu diễn words/tokens dưới dạng vectors có ý nghĩa**

### 🧭 Ví dụ trực quan
```
Word: "king"
Embedding: [0.2, -0.5, 0.8, 0.1, -0.3, ...]  (vector 768 chiều)

Tương tự:
"queen": [0.3, -0.4, 0.7, 0.2, -0.2, ...]
"man":   [0.1, -0.6, 0.2, 0.4, -0.1, ...]
"woman": [0.2, -0.3, 0.3, 0.5, 0.0, ...]
```

### 🧮 Toán học đằng sau
```
king - man + woman ≈ queen
[0.2,-0.5,0.8] - [0.1,-0.6,0.2] + [0.2,-0.3,0.3] ≈ [0.3,-0.4,0.7]
```

### 📈 Evolution của Embeddings

#### 1. **Word2Vec (2013)**
- Static embeddings: mỗi từ có 1 vector cố định
- "bank" luôn có cùng 1 embedding (dù là ngân hàng hay bờ sông)

#### 2. **GloVe (2014)**
- Tương tự Word2Vec nhưng học từ co-occurrence statistics

#### 3. **Contextual Embeddings (2018+)**
- BERT, GPT: embedding thay đổi theo context
- "bank" có embedding khác nhau tùy theo câu:
  - "I went to the **bank** to withdraw money" (ngân hàng)
  - "I sat by the river **bank**" (bờ sông)

### 🎯 Token Limits và Context Window
```
GPT-3.5: 4,096 tokens (~3,000 từ)
GPT-4:   8,192 tokens (~6,000 từ)
GPT-4-32k: 32,768 tokens (~24,000 từ)
Claude-2: 100,000 tokens (~75,000 từ)
```

**Tại sao có giới hạn?**
- Memory: Attention computation là O(n²)
- Cost: Xử lý nhiều tokens = đắt hơn
- Quality: Context quá dài có thể làm giảm focus

---

## 📚 Phần 3: LLM Parameters - Điều khiển "tính cách" của AI

### 🌡️ Temperature - Điều khiển Creativity

**Temperature = Mức độ "ngẫu nhiên" trong lựa chọn từ tiếp theo**

```
Prompt: "The weather today is"

Temperature = 0.0 (Deterministic):
"The weather today is sunny and warm."
"The weather today is sunny and warm." (luôn giống nhau)

Temperature = 0.3 (Conservative):
"The weather today is pleasant."
"The weather today is beautiful."

Temperature = 0.7 (Balanced):
"The weather today is absolutely gorgeous!"
"The weather today is a bit unpredictable."

Temperature = 1.0 (Creative):
"The weather today is dancing with possibilities!"
"The weather today is whispering secrets to the clouds."

Temperature = 2.0 (Chaotic):
"The weather today is purple elephant mathematics!"
```

### 🎯 Top-k Sampling - Giới hạn lựa chọn

**Top-k = Chỉ xem xét k từ có xác suất cao nhất**

```
Possible next words với probabilities:
"beautiful": 0.4
"sunny": 0.3  
"cloudy": 0.2
"rainy": 0.05
"snowy": 0.03
"terrible": 0.02

Top-k = 2: Chỉ chọn từ ["beautiful", "sunny"]
Top-k = 5: Chọn từ tất cả trừ "terrible"
```

### 🎪 Top-p (Nucleus) Sampling - Dynamic vocabulary

**Top-p = Chọn từ đến khi tổng xác suất ≥ p**

```
Top-p = 0.9:
"beautiful" (0.4) + "sunny" (0.3) + "cloudy" (0.2) = 0.9 ✓
→ Chọn từ 3 từ này

Top-p = 0.7:
"beautiful" (0.4) + "sunny" (0.3) = 0.7 ✓
→ Chỉ chọn từ 2 từ này
```

### 🔄 Repetition Penalty
**Giảm xác suất của từ đã xuất hiện**

```
Without repetition penalty:
"The cat sat on the mat. The cat was happy. The cat..."

With repetition penalty = 1.2:
"The cat sat on the mat. It was content. This feline..."
```

### 🌱 Seed - Reproducibility
```python
# Cùng prompt + cùng parameters + cùng seed = cùng output
generate_text(
    prompt="Write a function",
    temperature=0.7,
    seed=42  # Luôn cho kết quả giống nhau
)
```

---

## 🛠️ Phần Thực Hành - Jupyter Notebook Demos

### Demo 1: Tokenization Comparison
```python
# So sánh các tokenizer khác nhau
from transformers import AutoTokenizer

# Load different tokenizers
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello world! This is a test of tokenization methods."

print("GPT-2 Tokenization:")
gpt2_tokens = gpt2_tokenizer.encode(text)
print(f"Tokens: {gpt2_tokens}")
print(f"Decoded: {[gpt2_tokenizer.decode([t]) for t in gpt2_tokens]}")

print("\nBERT Tokenization:")
bert_tokens = bert_tokenizer.encode(text)
print(f"Tokens: {bert_tokens}")
print(f"Decoded: {[bert_tokenizer.decode([t]) for t in bert_tokens]}")
```

### Demo 2: Token Counting
```python
def count_tokens(text, tokenizer_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokens = tokenizer.encode(text)
    return len(tokens)

# Test với different text lengths
texts = [
    "Hi",
    "Hello world!",
    "This is a longer sentence to test tokenization.",
    "This is a much longer paragraph that contains multiple sentences. It should demonstrate how token count increases with text length. We can use this to understand context window limitations."
]

for text in texts:
    token_count = count_tokens(text)
    print(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}' → {token_count} tokens")
```

### Demo 3: Embedding Visualization
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example words/phrases
texts = [
    "king", "queen", "man", "woman",
    "happy", "sad", "joyful", "depressed",
    "cat", "dog", "animal", "pet",
    "car", "vehicle", "transportation", "bike"
]

# Get embeddings
embeddings = model.encode(texts)

# Reduce dimensions for visualization
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

# Add labels
for i, txt in enumerate(texts):
    plt.annotate(txt, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.title("Word Embeddings Visualization (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, alpha=0.3)
plt.show()
```

### Demo 4: Parameter Effects
```python
import openai
# Hoặc sử dụng local model với ollama

def test_temperature_effects(prompt, temperatures=[0.0, 0.3, 0.7, 1.0]):
    """Test different temperature settings"""
    results = {}
    
    for temp in temperatures:
        # Simulate API call (replace với actual API)
        response = generate_text(
            prompt=prompt,
            temperature=temp,
            max_tokens=50,
            seed=42  # For reproducibility in demo
        )
        results[temp] = response
    
    return results

# Test prompt
prompt = "Write a creative story about a robot learning to cook:"

results = test_temperature_effects(prompt)

for temp, result in results.items():
    print(f"\nTemperature {temp}:")
    print(f"Output: {result}")
    print("-" * 50)
```

### Demo 5: Context Window Testing
```python
def test_context_limits(base_text, max_tokens=4096):
    """Test how models handle long context"""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Gradually increase text length
    multipliers = [1, 2, 5, 10, 20]
    
    for mult in multipliers:
        text = base_text * mult
        token_count = len(tokenizer.encode(text))
        
        status = "✅ OK" if token_count < max_tokens else "❌ Too long"
        print(f"Text length: {len(text)} chars, Tokens: {token_count}, Status: {status}")
        
        if token_count > max_tokens:
            print(f"Exceeded limit by {token_count - max_tokens} tokens")
            break

# Test với sample text
sample_text = "This is a sample sentence that we will repeat multiple times to test context window limits. "
test_context_limits(sample_text)
```

---

## 🎮 Hands-on Activities

### Activity 1: Tokenizer Detective
**Nhiệm vụ:** Tìm hiểu tại sao một số từ được tokenize lạ

```python
# Test với những từ thú vị
test_words = [
    "ChatGPT",           # Tên riêng
    "antiestablishment", # Từ dài
    "🚀",                # Emoji
    "café",              # Accented characters
    "hello123",          # Mixed alphanumeric
    "don't",             # Contractions
]

for word in test_words:
    tokens = tokenizer.encode(word)
    decoded = [tokenizer.decode([t]) for t in tokens]
    print(f"'{word}' → {decoded} ({len(tokens)} tokens)")
```

### Activity 2: Embedding Math
**Nhiệm vụ:** Verify famous word analogy: King - Man + Woman = Queen

```python
def word_analogy(model, word1, word2, word3):
    """Solve word1 - word2 + word3 = ?"""
    
    # Get embeddings
    emb1 = model.encode([word1])[0]
    emb2 = model.encode([word2])[0] 
    emb3 = model.encode([word3])[0]
    
    # Calculate result vector
    result_vector = emb1 - emb2 + emb3
    
    # Find closest word (simplified - in practice you'd search a vocabulary)
    candidates = ["queen", "princess", "lady", "woman", "girl", "mother"]
    similarities = []
    
    for candidate in candidates:
        candidate_emb = model.encode([candidate])[0]
        similarity = np.dot(result_vector, candidate_emb) / (
            np.linalg.norm(result_vector) * np.linalg.norm(candidate_emb)
        )
        similarities.append((candidate, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{word1} - {word2} + {word3} = ?")
    print("Top candidates:")
    for word, sim in similarities[:3]:
        print(f"  {word}: {sim:.3f}")

# Test the famous example
word_analogy(model, "king", "man", "woman")
```

### Activity 3: Parameter Playground
**Nhiệm vụ:** Tìm combination tốt nhất cho different tasks

```python
# Different tasks require different settings
tasks = {
    "code_generation": {
        "prompt": "Write a Python function to calculate fibonacci:",
        "ideal_params": {"temperature": 0.1, "top_p": 0.9}  # Low creativity
    },
    "creative_writing": {
        "prompt": "Write a fantasy story about a magical forest:",
        "ideal_params": {"temperature": 0.8, "top_p": 0.95}  # High creativity
    },
    "data_analysis": {
        "prompt": "Analyze this sales data and provide insights:",
        "ideal_params": {"temperature": 0.2, "top_p": 0.9}   # Factual
    }
}

def test_parameter_combinations(task_name, task_info):
    prompt = task_info["prompt"]
    ideal = task_info["ideal_params"]
    
    # Test different combinations
    temp_values = [0.1, 0.5, 0.9]
    top_p_values = [0.7, 0.9, 0.95]
    
    print(f"\nTesting task: {task_name}")
    print(f"Ideal parameters: {ideal}")
    print("-" * 50)
    
    for temp in temp_values:
        for top_p in top_p_values:
            # Simulate generation (replace with actual API call)
            result = f"Generated with temp={temp}, top_p={top_p}"
            
            # Calculate "quality score" (simplified)
            distance_from_ideal = abs(temp - ideal["temperature"]) + abs(top_p - ideal["top_p"])
            quality = max(0, 1 - distance_from_ideal)
            
            print(f"temp={temp}, top_p={top_p} → Quality: {quality:.2f}")

# Test all tasks
for task_name, task_info in tasks.items():
    test_parameter_combinations(task_name, task_info)
```

---

## 🧠 Key Takeaways

### 1. **Hiểu biết về Tokenization**
- BPE cân bằng giữa hiệu quả và tính linh hoạt
- Số lượng token ảnh hưởng trực tiếp đến chi phí và hiệu suất
- Các mô hình khác nhau có tokenizer khác nhau

### 2. **Sức mạnh của Embedding**
- Các từ có nghĩa tương tự nhóm lại với nhau trong không gian vector
- Embedding theo ngữ cảnh tốt hơn embedding tĩnh
- Chiều embedding: lớn hơn = biểu cảm nhiều hơn nhưng tốn kém hơn

### 3. **Thành thạo Parameters**
- **Temperature**: 0.0-0.3 cho các tác vụ thực tế, 0.7-1.0 cho các tác vụ sáng tạo
- **Top-p**: Thường 0.9-0.95 hoạt động tốt
- **Seed**: Thiết yếu cho kết quả có thể tái tạo trong kiểm thử

### 4. **Ứng dụng Thực tế**
- Tạo code: Temperature thấp + top-p cao
- Viết sáng tạo: Temperature cao + top-p cao  
- Phân tích dữ liệu: Temperature rất thấp + top-p trung bình
- Chatbot: Temperature trung bình + top-p cao

---

## 🔄 Next Steps

1. **Thử nghiệm** với các tokenizer khác nhau trên văn bản của bạn
2. **Trực quan hóa** embeddings của các thuật ngữ chuyên ngành
3. **Tinh chỉnh** parameters cho các trường hợp sử dụng cụ thể của bạn
4. **Giám sát** việc sử dụng token để tối ưu hóa chi phí

---

## 📚 Additional Resources

- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers)
- [OpenAI Parameters Guide](https://platform.openai.com/docs/api-reference/completions)
- [Sentence Transformers Library](https://www.sbert.net/)
- [BPE Paper](https://arxiv.org/abs/1508.07909)

---

*"Hiểu về tokenization và embeddings là nền tảng để thành thạo AI hiện đại. Parameters là công cụ để điều chỉnh hành vi AI theo nhu cầu của bạn."*
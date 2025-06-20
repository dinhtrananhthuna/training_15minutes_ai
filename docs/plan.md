# 15 Phút Với AI - Chương Trình Training Hằng Ngày

## Giới thiệu
Chương trình training "15 phút với AI" được thiết kế dành riêng cho các lập trình viên, tập trung vào kiến thức kỹ thuật và ứng dụng thực tế của AI trong lập trình. Mỗi buổi training diễn ra 15 phút sau daily standup.

---

## Tuần 1: Nền Tảng AI và Machine Learning

### Ngày 1: Giới thiệu AI, ML, DL và GenAI
**Chủ đề:** Phân biệt AI, Machine Learning, Deep Learning và Generative AI

**Nội dung:**
- AI vs ML vs DL: Mối quan hệ và sự khác biệt
- Traditional ML vs Deep Learning: Khi nào dùng gì?
- Generative AI: Từ GAN đến Transformer
- Các loại model: Discriminative vs Generative

**Demo:**
```python
# Jupyter Notebook Demo
import numpy as np
import matplotlib.pyplot as plt

# Ví dụ về Traditional ML vs Deep Learning
# Traditional ML: Linear Regression
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10])

# Deep Learning approach sẽ cần nhiều layers
# Sẽ demo bằng simple neural network
```

**Cursor AI Demo:** Sử dụng Cursor để generate code comparison giữa traditional ML và neural network

---

### Ngày 2: Transformer Architecture Deep Dive
**Chủ đề:** Hiểu sâu về Transformer - backbone của LLM

**Nội dung:**
- Self-Attention mechanism: Query, Key, Value
- Multi-Head Attention: Tại sao cần nhiều heads?
- Positional Encoding: Làm thế nào model hiểu thứ tự?
- Feed Forward Networks trong Transformer

**Demo:**
```python
# Implement simple attention mechanism
import torch
import torch.nn as nn

class SimpleAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        attention_weights = torch.softmax(Q @ K.T / (self.d_model ** 0.5), dim=-1)
        output = attention_weights @ V
        return output
```

**Cursor AI Demo:** Generate và explain transformer components

---

### Ngày 3: Tokenization và Embedding
**Chủ đề:** Làm thế nào máy tính hiểu được text?

**Nội dung:**
- Byte-Pair Encoding (BPE): Thuật toán tokenization phổ biến
- SentencePiece vs WordPiece vs BPE
- Token limits: Tại sao GPT-4 có 8K/32K tokens?
- Embedding spaces: Word2Vec, GloVe, contextual embeddings

**Demo:**
```python
# Tokenization demo
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Hello world! This is a test."
tokens = tokenizer.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Decoded: {tokenizer.decode(tokens)}")

# Visualize token boundaries
for i, token_id in enumerate(tokens):
    token = tokenizer.decode([token_id])
    print(f"Token {i}: '{token}' (ID: {token_id})")
```

**Cursor AI Demo:** Sử dụng Cursor để explore different tokenizers

---

### Ngày 4: Parameters trong LLM - Temperature, Top-k, Top-p
**Chủ đề:** Kiểm soát output của LLM như thế nào?

**Nội dung:**
- Temperature: Creativity vs Consistency
- Top-k sampling: Giới hạn vocabulary
- Top-p (nucleus) sampling: Dynamic vocabulary selection
- Repetition penalty: Tránh lặp từ
- Seed: Reproducible outputs

**Demo:**
```python
# Demo với OpenAI API hoặc local model
import openai

prompts = ["Write a function to sort an array"]

# Different temperature settings
for temp in [0.1, 0.7, 1.0, 1.5]:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompts[0]}],
        temperature=temp,
        max_tokens=100
    )
    print(f"Temperature {temp}:")
    print(response.choices[0].message.content)
    print("-" * 50)
```

**Cursor AI Demo:** Thử nghiệm các parameters khác nhau trong Cursor settings

---

### Ngày 5: Context Window và Memory Management
**Chủ đề:** Làm thế nào LLM "nhớ" cuộc hội thoại?

**Nội dung:**
- Context window: 4K, 8K, 32K, 128K tokens
- Sliding window attention: Longformer, BigBird
- Memory mechanisms: Recurrent memory, external memory
- Context compression techniques

**Demo:**
```python
# Demonstrate context window limits
def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

# Test với different context lengths
long_text = "This is a test. " * 1000
print(f"Token count: {count_tokens(long_text, tokenizer)}")

# Strategies for long context
def sliding_window_context(text, window_size=2000):
    tokens = tokenizer.encode(text)
    if len(tokens) <= window_size:
        return text
    
    # Keep first and last parts
    first_half = tokens[:window_size//2]
    last_half = tokens[-(window_size//2):]
    
    return tokenizer.decode(first_half + last_half)
```

**Cursor AI Demo:** Test context limits với Cursor chat

---

## Tuần 2: Practical AI Tools cho Lập Trình

### Ngày 6: Cursor AI Deep Dive
**Chủ đề:** Maximize productivity với Cursor AI

**Nội dung:**
- Cursor AI architecture và capabilities
- Best practices: Comment-driven development
- Advanced features: Composer, Chat, Codebase indexing
- Customization: Rules và instructions cho team

**Demo:**
- Live coding session với Cursor AI
- Multi-file editing với Composer
- Codebase search và context understanding
- Custom rules setup cho team workflow

**Practice:** Refactor một module lớn bằng Cursor AI

---

### Ngày 7: Cursor AI Advanced Features
**Chủ đề:** Tận dụng tối đa Cursor AI

**Nội dung:**
- Cursor Composer: Multi-file editing
- Custom instructions và rules
- Codebase indexing và search
- Integration với existing workflow

**Demo:**
- Refactor một module lớn bằng Composer
- Setup custom rules cho team
- Debug session với Cursor AI

---

### Ngày 8: Prompt Engineering Fundamentals
**Chủ đề:** Zero-shot và Few-shot Prompting

**Nội dung:**
- Zero-shot prompting: Direct task instruction
- Few-shot prompting: Learning from examples
- Role-based prompting: System personas
- Context setting và instruction clarity

**Demo:**
```python
# Zero-shot prompting
zero_shot = """
You are a senior Python developer. Write a function to find the maximum value in a nested dictionary.
Include error handling, type hints, and docstring.
"""

# Few-shot prompting
few_shot = """
Here are examples of good Python functions:

Example 1:
def calculate_average(numbers: List[float]) -> float:
    \"\"\"Calculate the average of a list of numbers.\"\"\"
    if not numbers:
        raise ValueError("List cannot be empty")
    return sum(numbers) / len(numbers)

Example 2:
def find_duplicates(items: List[str]) -> Set[str]:
    \"\"\"Find duplicate items in a list.\"\"\"
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return duplicates

Now write a similar function to merge two sorted lists.
"""

# Role-based prompting
role_based = """
You are a code reviewer with 10 years of experience. 
Review this code for:
- Performance issues
- Security vulnerabilities  
- Best practices violations
- Maintainability concerns

[CODE TO REVIEW]
"""
```

**Cursor AI Demo:** Test different prompting approaches

---

### Ngày 9: Advanced Prompt Patterns
**Chủ đề:** Chain of Thought, Tree of Thought và Interview Pattern

**Nội dung:**
- Chain of Thought (CoT): Step-by-step reasoning
- Tree of Thought (ToT): Multiple reasoning paths
- Interview Pattern: Iterative questioning
- Self-consistency và verification patterns

**Demo:**
```python
# Chain of Thought prompting
cot_prompt = """
Optimize this SQL query step by step:

SELECT * FROM users u 
JOIN orders o ON u.id = o.user_id 
WHERE u.created_at > '2023-01-01' 
AND o.total > 100

Let me think through this step by step:
1. First, I'll analyze the current query structure
2. Then, I'll identify potential performance bottlenecks
3. Next, I'll suggest specific optimizations
4. Finally, I'll provide the optimized query

Step 1: Current query analysis...
"""

# Tree of Thought prompting
tot_prompt = """
Design a caching strategy for this API. Consider multiple approaches:

Path A: Redis-based caching
- Pros: Fast, distributed
- Cons: Memory usage, complexity
- Implementation: [details]

Path B: In-memory caching
- Pros: Simple, fast
- Cons: Limited scalability
- Implementation: [details]

Path C: Database-level caching
- Pros: Persistent, consistent
- Cons: Slower than memory
- Implementation: [details]

Now evaluate each path and recommend the best approach.
"""

# Interview Pattern
interview_prompt = """
I want to implement user authentication. Ask me questions to understand my requirements better.

Start with: "What type of authentication do you need?"
Based on my answer, ask follow-up questions about:
- User types and roles
- Security requirements
- Integration needs
- Scalability concerns

Continue asking until you have enough information to provide a complete solution.
"""

# Self-consistency pattern
self_consistency = """
Solve this algorithm problem using 3 different approaches:

Problem: Find the longest palindromic substring

Approach 1: Brute force
[Your solution]

Approach 2: Dynamic programming
[Your solution]

Approach 3: Expand around centers
[Your solution]

Now compare the solutions and verify which is most efficient.
"""
```

**Cursor AI Demo:** Practice advanced patterns với real coding problems

---

### Ngày 10: Prompt Engineering Best Practices
**Chủ đề:** Optimization và Meta-prompting Techniques

**Nội dung:**
- Prompt optimization strategies
- Meta-prompting: Prompts that generate prompts
- Context window management
- Error handling và fallback strategies

**Demo:**
```python
# Meta-prompting example
meta_prompt = """
Generate a prompt that will help a developer write better unit tests.
The prompt should:
1. Encourage comprehensive test coverage
2. Include edge cases and error conditions
3. Follow testing best practices
4. Be specific to the programming language

Generate the prompt now:
"""

# Context-aware prompting
context_aware = """
Based on this codebase context:
- Language: Python
- Framework: FastAPI
- Database: PostgreSQL
- Testing: pytest
- Architecture: Microservices

Now help me implement authentication middleware.
Ensure your solution fits this specific tech stack.
"""

# Iterative refinement pattern
iterative_prompt = """
Initial request: "Create a REST API endpoint"

Refinement 1: "Create a REST API endpoint for user management"
Refinement 2: "Create a REST API endpoint for user management with CRUD operations"
Refinement 3: "Create a REST API endpoint for user management with CRUD operations, input validation, and error handling"

Final refined prompt: [Complete detailed specification]
"""

# Error handling pattern
error_handling = """
If you cannot complete the task as requested, please:
1. Explain what part you cannot do and why
2. Suggest alternative approaches
3. Ask clarifying questions
4. Provide a partial solution if possible

Example: "I cannot access external APIs, but I can show you how to structure the code for API integration."
"""
```

**Practice:** Create optimized prompts cho team's common tasks

---

### Ngày 11: Code Review với AI
**Chủ đề:** AI-assisted code review process

**Nội dung:**
- Automated code review tools
- Security vulnerability detection
- Performance optimization suggestions
- Code style và best practices

**Demo:**
- Setup AI code review trong GitHub Actions
- Review session với AI feedback
- Compare AI suggestions với human review

---

### Ngày 12: Testing và Debugging với AI
**Chủ đề:** AI trong QA và debugging process

**Nội dung:**
- Automated test generation
- Bug detection và root cause analysis
- Performance profiling với AI
- Test data generation

**Demo:**
```python
# AI-generated test cases
def generate_test_cases(function_code, ai_model):
    prompt = f"""
    Generate comprehensive test cases for this function:
    {function_code}
    
    Include:
    - Happy path tests
    - Edge cases
    - Error conditions
    - Performance tests
    """
    return ai_model.generate(prompt)
```

---

## Tuần 3: Advanced AI Concepts

### Ngày 13: Fine-tuning và Model Customization
**Chủ đề:** Customize AI models cho specific needs

**Nội dung:**
- Transfer learning vs Fine-tuning vs Training from scratch
- LoRA (Low-Rank Adaptation): Efficient fine-tuning
- Dataset preparation và quality
- Evaluation metrics

**Demo:**
```python
# Fine-tuning example với Hugging Face
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Setup training arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    warmup_steps=500,
    logging_steps=10,
)
```

---

### Ngày 14: Retrieval Augmented Generation (RAG)
**Chủ đề:** Enhance LLM với external knowledge

**Nội dung:**
- RAG architecture: Retriever + Generator
- Vector databases: Pinecone, Weaviate, Chroma
- Embedding models: Sentence-BERT, OpenAI embeddings
- Chunking strategies và similarity search

**Demo:**
```python
# Simple RAG implementation
import chromadb
from sentence_transformers import SentenceTransformer

# Setup vector database
client = chromadb.Client()
collection = client.create_collection("codebase")

# Embed và store code snippets
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def add_code_snippet(code, description):
    embedding = embedder.encode([description])
    collection.add(
        embeddings=embedding.tolist(),
        documents=[code],
        metadatas=[{"description": description}],
        ids=[f"snippet_{len(collection.get()['ids'])}"]
    )

def search_similar_code(query):
    query_embedding = embedder.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )
    return results
```

---

### Ngày 15: Local LLM Setup và Optimization
**Chủ đề:** Chạy LLM trên local machine

**Nội dung:**
- Ollama, LM Studio, GPT4All
- Model quantization: GGUF, GPTQ
- Hardware requirements: CPU vs GPU
- Privacy và security considerations

**Demo:**
- Setup Ollama với Code Llama
- Performance comparison: local vs cloud
- Integration với development workflow

---

### Ngày 16: AI Agents và Automation
**Chủ đề:** Building AI agents cho development tasks

**Nội dung:**
- Agent frameworks: LangChain, AutoGPT, CrewAI
- Tool usage và function calling
- Multi-agent systems
- Workflow automation

**Demo:**
```python
# Simple coding agent
from langchain.agents import create_openai_functions_agent
from langchain.tools import tool

@tool
def run_code(code: str) -> str:
    """Execute Python code and return the result"""
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return "Code executed successfully"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def write_file(filename: str, content: str) -> str:
    """Write content to a file"""
    with open(filename, 'w') as f:
        f.write(content)
    return f"File {filename} written successfully"

# Create agent với tools
agent = create_openai_functions_agent(
    llm=llm,
    tools=[run_code, write_file],
    prompt=coding_agent_prompt
)
```

---

### Ngày 17: Model Context Protocol (MCP) & External Integrations
**Chủ đề:** Kết nối AI với External Systems qua MCP

**Nội dung:**
- MCP là gì? Architecture Client-Server
- Setup MCP trong Cursor AI: Figma, GitHub, Filesystem
- Use cases thực tế: Design-to-Code workflow
- So sánh MCP vs Traditional APIs

**Demo:**
```json
// Cursor AI MCP configuration
{
  "mcpServers": {
    "figma": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-figma"],
      "env": {"FIGMA_ACCESS_TOKEN": "your-token"}
    },
    "github": {
      "command": "npx", 
      "args": ["@modelcontextprotocol/server-github"],
      "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "your-token"}
    }
  }
}
```

**Hands-on:**
- Setup Figma MCP server
- Connect GitHub repository
- Generate React components từ Figma designs
- Extract design tokens và sync với codebase

---

### Ngày 18: Advanced AI Development Patterns
**Chủ đề:** Patterns và best practices cho AI-driven development

**Nội dung:**
- AI-first development workflow
- Prompt patterns cho complex tasks
- Error handling với AI systems
- Performance optimization strategies

**Demo:**
```python
# Advanced prompt patterns
class AIAssistant:
    def __init__(self):
        self.context_window = []
        
    def chain_of_thought_prompt(self, task):
        prompt = f"""
        Task: {task}
        
        Let me think through this step by step:
        1. First, I need to understand the requirements
        2. Then, I'll break down the problem
        3. Next, I'll implement the solution
        4. Finally, I'll test and validate
        
        Step 1: Understanding requirements...
        """
        return self.generate(prompt)
    
    def few_shot_examples(self, task, examples):
        prompt = f"Here are some examples:\n"
        for ex in examples:
            prompt += f"Input: {ex['input']}\nOutput: {ex['output']}\n\n"
        prompt += f"Now solve: {task}"
        return self.generate(prompt)
```

**Cursor AI Demo:** Apply advanced patterns trong real project

---

### Ngày 19: AI Security & Ethics
**Chủ đề:** Responsible AI development và security considerations

**Nội dung:**
- AI security vulnerabilities: Prompt injection, data leakage
- Code review với AI: Security best practices
- Privacy considerations khi dùng AI tools
- Ethical AI development guidelines

**Demo:**
```python
# Security-aware AI usage
class SecureAIHelper:
    def __init__(self):
        self.sensitive_patterns = [
            r'password\s*=\s*["\'].*["\']',
            r'api_key\s*=\s*["\'].*["\']',
            r'secret\s*=\s*["\'].*["\']'
        ]
    
    def sanitize_input(self, code):
        for pattern in self.sensitive_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return "⚠️ Sensitive data detected. Please remove before processing."
        return code
    
    def secure_prompt(self, user_input):
        # Prevent prompt injection
        sanitized = self.sanitize_input(user_input)
        return f"System: Process this code safely:\n{sanitized}"
```

**Discussion:** Team guidelines cho responsible AI usage

---

### Ngày 20: Performance Optimization với AI
**Chủ đề:** Optimize code performance bằng AI tools

**Nội dung:**
- AI-powered code profiling
- Performance bottleneck detection
- Automated optimization suggestions
- Benchmarking và monitoring

**Demo:**
```python
# AI-assisted performance optimization
def optimize_with_ai(code_snippet):
    analysis_prompt = f"""
    Analyze this code for performance issues:
    {code_snippet}
    
    Provide:
    1. Performance bottlenecks
    2. Time complexity analysis
    3. Optimization suggestions
    4. Refactored code
    """
    
    # Get AI analysis
    analysis = ai_model.generate(analysis_prompt)
    
    # Apply optimizations
    optimized_code = extract_optimized_code(analysis)
    
    return {
        'analysis': analysis,
        'optimized_code': optimized_code,
        'performance_gain': benchmark_comparison(code_snippet, optimized_code)
    }
```

**Cursor AI Demo:** Optimize một function có performance issues

---

### Ngày 21: AI trong DevOps & CI/CD
**Chủ đề:** Integrate AI vào development lifecycle

**Nội dung:**
- AI trong automated testing
- Intelligent deployment strategies
- Log analysis với AI
- Predictive maintenance

**Demo:**
```yaml
# GitHub Actions với AI
name: AI-Powered CI/CD
on: [push, pull_request]

jobs:
  ai-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: AI Code Review
        run: |
          # Sử dụng AI để review code changes
          ai-reviewer --diff="$(git diff HEAD~1)" \
                      --output="review-comments.md" \
                      --severity="medium"
      
      - name: AI Test Generation
        run: |
          # Generate tests cho new functions
          ai-test-gen --source="src/" \
                      --output="tests/ai-generated/" \
                      --coverage-target="80%"
```

**Practice:** Setup AI-powered GitHub Actions

---

### Ngày 22: Future of AI in Software Development
**Chủ đề:** Xu hướng và roadmap cho team

**Nội dung:**
- Emerging AI technologies: Multimodal AI, Code LLMs
- AI-first development workflow
- Building AI-native applications
- Team adoption roadmap

**Demo:**
- Prototype một AI-powered development tool
- Discussion về impact của AI lên software engineering
- Create team roadmap cho AI adoption
- Contribution guidelines cho AI tools

**Retrospective:**
- Review 22 ngày training
- Identify key takeaways
- Plan next steps cho team
- Setup continuous learning process

---

## Hướng Dẫn Thực Hiện

### Chuẩn Bị Trước Mỗi Buổi:
1. **Setup môi trường:**
   - Jupyter Notebook hoặc Google Colab
   - Cursor AI với latest version
   - GitHub Copilot (nếu có)

2. **Materials cần thiết:**
   - Code examples đã chuẩn bị
   - Demo scripts
   - Relevant documentation links

### Format Mỗi Buổi Training (15 phút):
- **Phút 1-2:** Quick recap và giới thiệu topic
- **Phút 3-10:** Core content với live demo
- **Phút 11-14:** Hands-on practice hoặc Q&A
- **Phút 15:** Wrap-up và preview ngày mai

### Tips cho Trainer:
1. **Keep it interactive:** Encourage questions và participation
2. **Real examples:** Sử dụng code từ actual projects
3. **Progressive complexity:** Build từ basic đến advanced
4. **Practical focus:** Always connect theory với real-world usage

### Resources và Tools:
- **Jupyter Notebooks:** Cho interactive demos
- **Cursor AI:** Primary coding assistant
- **GitHub:** Code examples và repositories
- **Online playgrounds:** Hugging Face Spaces, Google Colab
- **Documentation:** Official docs của các tools/models

### Follow-up Activities:
- **Weekly recap:** Summary của 5 ngày training
- **Hands-on assignments:** Optional coding challenges
- **Team sharing:** Members share their AI discoveries
- **Tool evaluation:** Regular assessment của new AI tools

### MCP Setup Guide:
1. **Prerequisites:**
   - Node.js 18+ installed
   - Cursor AI latest version
   - API tokens (Figma, GitHub, etc.)

2. **Quick Start:**
   ```bash
   # Install MCP servers
   npm install -g @modelcontextprotocol/server-figma
   npm install -g @modelcontextprotocol/server-github
   npm install -g @modelcontextprotocol/server-filesystem
   ```

3. **Cursor AI Configuration:**
   - Open Cursor Settings (Cmd/Ctrl + ,)
   - Navigate to "Features" → "Model Context Protocol"
   - Add server configurations:
   ```json
   {
     "mcpServers": {
       "figma": {
         "command": "npx",
         "args": ["@modelcontextprotocol/server-figma"],
         "env": {
           "FIGMA_ACCESS_TOKEN": "figd_your_token_here"
         }
       }
     }
   }
   ```

4. **Testing MCP Connection:**
   - Restart Cursor AI
   - Open chat và type: "@figma get file information"
   - Verify MCP server responds correctly

---

## Appendix: Useful Resources

### Documentation:
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

### Tools:
- [Cursor AI](https://cursor.sh/)
- [GitHub Copilot](https://github.com/features/copilot)
- [Ollama](https://ollama.ai/)
- [LangChain](https://langchain.com/)

### MCP Resources:
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [MCP SDK Documentation](https://github.com/modelcontextprotocol/sdk)
- [Official MCP Servers](https://github.com/modelcontextprotocol/servers)
- [Figma MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/figma)
- [GitHub MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/github)

### Communities:
- [r/MachineLearning](https://reddit.com/r/MachineLearning)
- [AI/ML Twitter Community](https://twitter.com/search?q=%23MachineLearning)
- [Hugging Face Community](https://huggingface.co/community)

---

*Chương trình này được thiết kế để team có thể áp dụng ngay kiến thức vào công việc hằng ngày. Mỗi concept đều có practical application và demo code cụ thể.* 
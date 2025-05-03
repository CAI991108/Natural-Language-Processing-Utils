# Natural Language Processing Utils Repository 🚀

![NLP Wizard](https://luminoso.com/wp-content/uploads/2023/11/Definitive-Guide-to-Natural-Language-Processing.png)  

Welcome to my NLP assignments repository! Here, you'll find three projects exploring cutting-edge NLP techniques—from **word embeddings** to **LLM fine-tuning** and **medical agent frameworks**. Each assignment tackles real-world challenges with code, experiments, and detailed reports. Let’s dive in! 🔍

---

## 📂 Assignment Overview

| Assignment | Topics Covered | Key Techniques | Dataset/Model | Highlights |
|------------|----------------|----------------|----------------|------------|
| [**1: Word Embeddings**](NLP_Assignment_1.pdf) | • Word2Vec/CBoW<br>• Embedding visualization<br>• Polysemy & analogies | • Skip-gram vs CBoW<br>• Cosine similarity<br>• t-SNE projections | Quora Question Pairs | Trained embeddings from scratch, achieved semantic clustering for words like *oil* and *banking*! |
| [**2: LLM Medical Agent**](NLP_Assignment_2.pdf) | • Prompt engineering<br>• Multi-agent validation<br>• Regulatory compliance | • Chain-of-Thought<br>• Self-consistency checks<br>• Tavily API integration | National Pharmacist Exam Qs | Boosted accuracy by **15%** using agent debates and real-time guideline checks! 🩺✅ |
| [**3: LLM Fine-Tuning**](NLP_Assignment_3.pdf) | • QLoRA/PEFT<br>• Chinese medical QA<br>• 4-bit quantization | • LoRA adapters<br>• Instruction tuning<br>• Perplexity evaluation | Huatuo26M-Lite<br>Qwen-7B-Instruct | Achieved **80% accuracy** on medical MCQs with just **12.76GB VRAM**! 🌟 |

---

## 🛠️ Technical Deep-Dive

### Assignment 1: Word Embeddings
- **Core Idea**: "You shall know a word by the company it keeps!"  
- **Cool Finds**:  
  - `oil` clustered with *ecuador* and *industry* but not *venezuela* 🤔  
  - **Bias alert**: `rich` was closer to `poor` than `affluent` in cosine space!  
- **Tools**: PyTorch, NLTK, Gensim, Bokeh.  
- [Code Snippets](https://github.com/NLP-Course-CUHKSZ/NLP-course-cuhksz.github.io/tree/main/Assignments/Assignment1)

---

### Assignment 2: Medical LLM Agent
- **Innovation**: Multi-agent debate framework:  
  - **Drug Interaction Agent** 🧪 + **Regulatory Agent** 📜 → Consensus-driven answers.  
- **Result**: 74.3% accuracy on best-choice questions with **<5s latency**.  
- **Try This Prompt**:  
  ```python
  "As a pharmacist, cross-reference [Drug X] with [2024 formulary] before answering."
  ```
* [Agent Code](https://github.com/PawkyFox/Prompt-Engineering-Guide/blob/main/README.md)

---

### Assignment 3: QLoRA Fine-Tuning
- **Breakthrough**: Supercharged Qwen-7B for Chinese medical QA with **4-bit quantization** 🧠💡  
- **Tech Stack**: HuggingFace Transformers, PEFT, Accelerate.
- **Key Innovation**:  
  - Trained on **Huatuo26M-Lite** (doctor-patient dialogues) using **PEFT** and LoRA adapters.  
  - Achieved **80% accuracy** on medical MCQs while sipping just **12.76GB VRAM**!  
- **Sample Output**:  
  > **Patient**: *"喉咙不舒服，瓜子吃多了怎么办？"*  
  > **Model**:  
  > 1. 多喝水保持咽喉湿润  
  > 2. 停止食用瓜子  
  > 3. 含服润喉片  
  > 4. 症状持续需就医 → **临床指南对齐** ✅  
- **Tech Stack**:  
  ```python
  # QLoRA Config
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16
  )
  model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", 
              quantization_config=bnb_config)
  ```
- **Evaluation**:

    - Perplexity **↓5.31** (vs baseline >10)

    - Peak GPU usage: **12.76GB** 🚀

- [Colab Demo](https://colab.research.google.com/drive/1AP5kIzo_BKqfmmY8p1412kkNec82kMoh) | [HuggingFace Model](https://huggingface.co/Qwen)

---

## 📊 Results at a Glance
| Metric                | Assignment 1          | Assignment 2          | Assignment 3          |
|-----------------------|-----------------------|-----------------------|-----------------------|
| **Accuracy**          | N/A                   | 74.3%                 | 80%                   |
| **Training Time**     | 2.5 hours             | API-based             | 8 hours               |
| **GPU Memory Usage**  | 8GB (T4)              | N/A                   | 12.76GB               |
| **Key Visualization** | t-SNE embeddings      | Compliance heatmaps   | Perplexity=5.31 📉    |

---

## 🚀 How to Replicate
1. **Setup Environment**:  
   ```bash
   conda create -n nlp_course python=3.9
   conda install pytorch=2.0 cudatoolkit=11.7 -c pytorch
   pip install -r requirements.txt
   ```

2. **Run Experiments**:
    - Assignment 1 (Word2Vec):
    ```bash
    python train_cbow.py --window_size 2 --batch_size 128
    ```
    - Assignment 2 (Medical Agent):
    ```bash
    python agent_orchestrator.py --use_debate True --api_key YOUR_API_KEY
    ```
    - Assignment 3 (QLoRA Fine-Tuning):
    ```bash
    accelerate launch finetune_medical.py --model Qwen-7B --dataset Huatuo26M-Lite
    ```
3. **Evaluate & Visualize**:

    - Generate t-SNE plots for embeddings (Assignment 1).

    - Check compliance logs in `data/wrong_answers.json` (Assignment 2).

    - Monitor training metrics with `tensorboard --logdir ./logs`(Assignment 3).

---
## 📚 Resources & References
- **Guides & Tools**:

    🔗 [Prompt Engineering Guide](https://www.promptingguide.ai/)

    🔗 [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft)

    🔗 [LangChain Agent Framework](https://python.langchain.com/docs/introduction/)

- **Datasets & Models**:

    📂 [Huatuo26M-Lite Dataset](https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite)

    🤖 [Qwen-7B Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

    🧪 [Tavily API for Real-Time Validation](https://app.tavily.com/home)

- **Code Repos**:

    ⚙️ [Assignment 1 Code](https://github.com/NLP-Course-CUHKSZ/NLP-course-cuhksz.github.io/tree/main/Assignments/Assignment1)

    ⚙️ [Medical Agent Framework](https://github.com/PawkyFox/Prompt-Engineering-Guide)

---

Made with ❤️ by Zijin CAI 

*"From embeddings to life-saving diagnostics—code that cares!"* 🌟
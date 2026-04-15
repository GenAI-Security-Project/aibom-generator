# 🤖 OWASP GenAI Security Project - AIBOM Generator

This is the official GitHub repository for the **OWASP AIBOM Generator** — an open-source tool for generating **AI Bills of Materials (AIBOMs)** in [CycloneDX](https://cyclonedx.org) format.  
The tool is also listed in the official **[CycloneDX Tool Center](https://cyclonedx.org/tool-center/)**.

🚀 **Try the tool live:**  
👉 https://owasp-genai-aibom.org  
🔖 Bookmark and share: https://owasp-genai-aibom.org 

🌐 OWASP AIBOM Initiative: [genai.owasp.org](https://genai.owasp.org/)

> This initiative is about making AI transparency practical. The OWASP AIBOM Generator, running under the OWASP GenAI Security Project, is focused on helping organizations understand what’s actually inside AI models and systems, starting with open models on Hugging Face.
> Join OWASP GenAI Security Project - AIBOM Initiative to contribute.

---

## 📦 What It Does

- Extracts metadata from models hosted on Hugging Face 🤗  
- Generates an **AIBOM** (AI Bill of Materials) in CycloneDX 1.6 JSON format  
- Calculates **AIBOM completeness scoring** with recommendations  
- Supports metadata extraction from model cards, configurations, and repository files  

---

## 🛠 Features

- Human-readable AIBOM viewer  
- JSON download  
- Completeness scoring & improvement tips  
- API endpoints for automation  
- Standards-aligned generation (CycloneDX 1.6, compatible with SPDX AI Profile)

---

## � Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Or, if you prefer [uv](https://docs.astral.sh/uv/) for faster dependency management:
```bash
uv sync
```

### 2. Run Web Application
Start the local server at `http://localhost:8000`:
```bash
python3 -m src.main
```

### 3. Run via CLI
Generate an AIBOM for a Hugging Face model directly from your terminal:

**Basic Usage:**
```bash
python3 -m src.cli google-bert/bert-base-uncased
```

**Advanced Usage:**
You can specify additional metadata like component name, version, and supplier.
```bash
python3 -m src.cli google-bert/bert-base-uncased \
  --name "My Custom BERT" \
  --version "1.0.0" \
  --manufacturer "Acme Corp" \
  --output "my_sbom.json"
```

**Command Line Options:**

| Option | Shorthand | Description |
|--------|-----------|-------------|
| `model_id` | | Hugging Face Model ID (e.g., `owner/model`) |
| `--test` | `-t` | Run test mode for multiple predefined models |
| `--output` | `-o` | Custom output file path |
| `--name` | `-n` | Override component name in metadata |
| `--version` | `-v` | Override component version in metadata |
| `--manufacturer` | `-m` | Override component manufacturer/supplier |
| `--inference` | `-i` | Use AI inference for enhanced metadata (requires API key) |
| `--summarize` | `-s` | Enable intelligent description summarization |
| `--verbose` | | Enable verbose logging |

*   Metrics and produced SBOMs are saved to the `sboms/` directory by default.

---

## �🐞 Found a Bug or Have an Improvement Request?

We welcome contributions and feedback.

➡ **Log an issue:**  
https://github.com/GenAI-Security-Project/aibom-generator/issues

---

## 📄 License

This project is open-source and available under the [Apache 2.0 License](LICENSE).

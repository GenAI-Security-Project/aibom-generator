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

> **Built on [CycloneDX Python Library](https://github.com/CycloneDX/cyclonedx-python-lib)** — BOM generation, schema validation, and SPDX license handling are all powered by the official `cyclonedx-python-lib`, ensuring spec-compliant output without manual JSON construction.

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

### 4. Optional: candidate OWASP GenAI risks and mapped controls

Off by default. Any of the options below adds the OWASP GenAI risks that apply
to this *kind* of model, and the framework controls mapped to each, from the
[OWASP GenAI Crosswalk](https://github.com/GenAI-Security-Project/crosswalk).
These are **candidates, not an assessment**: nothing says the model has a risk
or meets a control, and every control carries the crosswalk's review status
(`reviewed`, `draft` or `unreviewed`). The completeness score is not affected.
In the web app, the same choices are under *Optional: add candidate OWASP GenAI
risks* on the form, and the result page gets a **Risks** tab.

```bash
python3 -m src.cli openai-community/gpt2 --owasp-risks
python3 -m src.cli openai-community/gpt2 --agentic --rag --frameworks "OWASP AISVS 1.01,NIST AI RMF 1.0"
```

| Option | Adds |
|--------|------|
| `--owasp-risks` | The section; the LLM Top 10 applies when the model's task is generative |
| `--llm` | The LLM Top 10, whatever the task |
| `--agentic` | The Agentic Top 10 (the model runs in an agent) |
| `--rag`, `--training-data` | GenAI Data Security risks (retrieval, or organisational training data) |
| `--skills` | The Agentic Skills Top 10 (the model loads agent skills) |
| `--frameworks` | Comma-separated framework ids or names; default is the crosswalk's verified core |

Where it goes in the BOM:

- `components[0].modelCard.considerations.ethicalConsiderations[]`: one
  `{name, mitigationStrategy}` per candidate risk, the strategy starting
  "Candidate risk — not assessed."
- `definitions.standards[]`: one standard per OWASP list and per framework.
  Control text is included only where the framework's licence allows it
  (ISO, PCI DSS, SOC 2, AIUC-1, ISA/IEC 62443 and CIS give an id and a link).
  Each control's `owasp:aibom:risk:link` properties read
  `<risk>|<relationship>|<review status>`.
- `metadata.properties`: `owasp:aibom:risk:*` for the status
  (`candidate-not-assessed`), the crosswalk commit, the profiles and frameworks.

The data is vendored in `src/data/owasp_crosswalk/` at a pinned crosswalk
commit (`SOURCE.json`) and is never fetched at run time. To update it:
`python scripts/update_crosswalk.py <crosswalk commit SHA>`. The data is
CC BY-SA 4.0; the BOM carries the attribution in `owasp:aibom:risk:attribution`.

---

## �🐞 Found a Bug or Have an Improvement Request?

We welcome contributions and feedback.

➡ **Log an issue:**  
https://github.com/GenAI-Security-Project/aibom-generator/issues

---

## 📄 License

This project is open-source and available under the [Apache 2.0 License](LICENSE).

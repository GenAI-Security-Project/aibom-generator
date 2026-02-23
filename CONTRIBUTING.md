# Contributing to OWASP GenAI Security Project - AIBOM Generator

Thank you for your interest in contributing to the OWASP AIBOM Generator! This project is part of the [OWASP GenAI Security Project](https://genai.owasp.org) and welcomes contributions from the community.

## Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/aibom-generator.git
   cd aibom-generator
   ```
3. **Set up your environment**:

   ```bash
   # Using Docker (recommended)
   docker build -t aibom .
   docker run -p 7860:7860 aibom

   # Or local Python setup
   pip install -r requirements.txt
   ```

## Development Workflow

### Branch Naming

Use descriptive branch names following this pattern:

```
type/issue-number-description
```

**Types:**

- `feat` - New features
- `fix` - Bug fixes
- `docs` - Documentation changes
- `refactor` - Code refactoring
- `test` - Adding or updating tests

**Examples:**

- `feat/17-schema-validation`
- `fix/13-purl-encoding`
- `docs/contributing-guide`

### Pull Request Process

1. Create a branch from `v0.2` (or `main` for documentation)
2. Make your changes with clear, focused commits
3. Push to your fork and open a PR
4. Link your PR to any related issues
5. Respond to review feedback

### Commit Messages

Use conventional commits:

```
type(scope): description

[optional body]
```

**Examples:**

- `feat(validation): add CycloneDX 1.6 schema validation`
- `fix(generator): correct PURL encoding for model IDs`

## Code Standards

### Python Style

- **Python 3.8+** compatibility required
- Follow existing patterns in the codebase
- Use type hints for function signatures

### Import Organization

```python
# Standard library
import json
import logging

# Third-party
import requests
from huggingface_hub import HfApi

# Local imports
from .models.service import AIBOMService
from .utils.validation import validate_aibom
```

### Logging Conventions

Use the Python `logging` module with module-level loggers:

```python
import logging

logger = logging.getLogger(__name__)

# Use lazy formatting (not f-strings)
logger.info("Processing model: %s", model_id)
logger.warning("Schema validation found %d issues", count)
logger.error("Failed to fetch model: %s", error, exc_info=True)
```

## Project Architecture (v0.2)

```
aibom-generator/
├── src/
│   ├── main.py               # Application entry point
│   ├── cli.py                # Command-line interface
│   ├── config.py             # Configuration settings
│   ├── controllers/
│   │   ├── cli_controller.py # CLI request handling
│   │   └── web_controller.py # Web/API request handling
│   ├── models/
│   │   ├── service.py        # Core AIBOM generation service
│   │   ├── extractor.py      # Metadata extraction
│   │   ├── registry.py       # Field registry management
│   │   ├── scoring.py        # Completeness scoring
│   │   └── schemas.py        # Pydantic models
│   ├── utils/
│   │   ├── validation.py     # CycloneDX 1.6 schema validation
│   │   ├── license_utils.py  # License normalization
│   │   └── analytics.py      # Usage tracking
│   ├── schemas/              # JSON schemas (CycloneDX, SPDX)
│   └── templates/            # HTML templates
├── tests/                    # Unit and integration tests
└── requirements.txt
```

### Key Concepts

- **Service-oriented architecture**: Core logic lives in `models/service.py`
- **Registry-driven fields**: Field definitions from `models/registry.py`
- **CycloneDX 1.6 compliance**: All AIBOMs validate against the schema
- **Completeness scoring**: Quality metrics in `models/scoring.py`

## Running Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_validation.py -v
```

## Building and Running

### Docker (Recommended)

```bash
docker build -t aibom .
docker run -p 7860:7860 aibom
```

### Local Development

```bash
pip install -r requirements.txt

# Run API server
python -m uvicorn src.main:app --reload --port 7860

# Or use CLI
python -m src.cli --model_id "microsoft/DialoGPT-medium"
```

## Areas Welcoming Contributions

We especially welcome contributions in these areas:

- **SPDX 3.1 export** - Add support for SPDX AI Profile format
- **Model-specific extractors** - Better metadata extraction for specific model types
- **Documentation** - Improve guides, examples, and API docs
- **UI/UX enhancements** - Improve the web interface
- **Test coverage** - Expand unit and integration tests
- **Performance optimization** - Faster extraction and generation

## Reporting Issues

Before creating an issue:

1. **Search existing issues** to avoid duplicates
2. Use a **clear, descriptive title**
3. For bugs, include:
   - Steps to reproduce
   - Expected vs actual behavior
   - Model ID if applicable
   - Error messages

**Issue tracker:** [GitHub Issues](https://github.com/GenAI-Security-Project/aibom-generator/issues)

## License

This project is licensed under the [Apache License 2.0](LICENSE). By contributing, you agree that your contributions will be licensed under the same license.

## Community

- **OWASP GenAI Security Project**: [genai.owasp.org](https://genai.owasp.org)
- **AIBOM Initiative**: [genai.owasp.org/ai-sbom-initiative](https://genai.owasp.org/ai-sbom-initiative)
- **Slack**: `#team-genai-aibom` on [owasp.slack.com](https://owasp.slack.com)
- **Weekly Call**: Mondays at 4:00 PM ET (Zoom: 851 1385 5952)

---

Thank you for helping make AI transparency practical!

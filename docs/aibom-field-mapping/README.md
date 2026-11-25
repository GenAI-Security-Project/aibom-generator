# AIBOM Field Mapping

## Purpose

This document defines a **shared, language-agnostic view of AI Bill of Materials (AIBOM) fields** and how they map to two commonly used AIBOM standardized formats:

- **[CycloneDX](https://cyclonedx.org/specification/overview/)**
- **[SPDX](https://spdx.dev/use/specifications/)**

The goal of this document is to provide an overview on:
- which fields are relevant for an AIBOM
- what each field means
- whether and how each field is supported in each format

This document intentionally avoids tool-specific implementation details.  
Those details live in a separate machine-readable mapping registry used by the generator.

---

## How to Read This Document

- **Language-agnostic fields** describe *what information is captured*, independent of AIBOM specifications (CDX/SPDX).
- The **coverage table** shows where each field is represented in CycloneDX and SPDX, using links to the relevant specification sections.
- Notes are used only where mapping is partial, or requires interpretation.

This document is meant for **overview and alignment**, not for prescribing internal tool architecture.

---

## Supported Languages

- **[CycloneDX 1.7](https://cyclonedx.org/docs/1.7/json/)**
- **[SPDX 3.0.1 (AI Profile)](https://spdx.github.io/spdx-spec/v3.0.1/model/AI/AI/)**


---

## Language-Agnostic Field Definitions

The table below defines the **core AIBOM fields**, independent of specification (CDX/SPDX).

These definitions are the foundation for both format mappings.

### Field Definitions

| Language-Agnostic Field | Description | Expected Data | Typical Source |
|------|-------------|---------------|----------------|
| Model name | Human-readable identifier of the AI model | string | Model metadata, model card |
| Model version | Version identifier for the model | string | Model metadata |
| Model description | Short narrative description of the model | text | Model card |
| Model download location | Where the model artifact can be retrieved | URL | Model registry |
| Source repository | Link to source code or training pipeline repository | URL | VCS |
| Supplier | Organization or individual providing the model | string | Publisher metadata |
| License | Legal license(s) governing the model | SPDX license ID / expression | Publisher metadata |
| Primary purpose | Intended task or use of the model | string | Model card |
| Domain | Domain(s) the model is intended for (e.g., NLP, CV) | list of strings | Model metadata |
| Type of model | Model family or learning type | string or list | Model documentation |
| Hyperparameters | Key training or inference parameters | key/value list | Training metadata |
| Performance metrics | Quantitative performance measurements | structured metrics | Evaluation results |
| Decision thresholds | Thresholds used for classification or decisions | structured values | Evaluation setup |
| Energy consumption | Energy usage related to training or inference | structured values | Training metadata |
| Training information | High-level description of how the model was trained | text | Model card |
| Data preprocessing | Preprocessing applied to training data | text or structured steps | Training pipeline |
| Sensitive data usage | Whether sensitive personal data was used | boolean / text | Model documentation |
| Explainability information | How model outputs can be interpreted | text | Model card |
| Known limitations | Known weaknesses or limitations | text | Model card |
| Safety risk assessment | Identified safety or risk considerations | text | Risk assessment |
| Runtime dependencies | Software dependencies required at runtime | list | Build / runtime metadata |
| Model lineage | Relationship to base or parent models | references | Model provenance |
| Training datasets | Datasets used during training | references | Dataset metadata |
| Test datasets | Datasets used for evaluation/testing | references | Dataset metadata |

---

## Field Coverage Across Formats

The table below shows how each format-agnostic field is represented in each specification.

Where possible, links point directly to the relevant specification section.

### Coverage Table

| Language-Agnostic Field | CycloneDX 1.7 | SPDX 3.0.1 (AI Profile) | Notes |
|------------------------|---------------|------------------------|-------|
| Model name | `component.name` | `AI.AIPackage.name` | Direct mapping |
| Model version | `component.version` | `packageVersion` | Direct mapping |
| Model description | `component.description` | `summary` | |
| Model download location | `externalReferences (distribution)` | `downloadLocation` | |
| Source repository | `externalReferences (vcs)` | `packageUrl` | |
| Supplier | `component.supplier` | `suppliedBy` | |
| License | `component.licenses` | License relationships | SPDX uses explicit relationships |
| Primary purpose | `modelCard.modelParameters.task` | `primaryPurpose` | |
| Domain | `component.tags` | `domain` | |
| Type of model | `modelCard.modelParameters.approach` | `typeOfModel` | |
| Hyperparameters | `modelCard.modelParameters` | `hyperparameter` | |
| Performance metrics | `modelCard.quantitativeAnalysis` | `metric` | |
| Decision thresholds | Partial support | `metricDecisionThreshold` | Limited in CDX |
| Energy consumption | `environmentalConsiderations` | `energyConsumption` | |
| Training information | `formulation` | `informationAboutTraining` | |
| Data preprocessing | `formulation` | `modelDataPreprocessing` | |
| Sensitive data usage | Partial | `useSensitivePersonalInformation` | |
| Explainability information | Extension | `modelExplainability` | |
| Known limitations | `technicalLimitations` | `limitation` | |
| Safety risk assessment | Partial | `safetyRiskAssessment` | |
| Runtime dependencies | `dependencies` | `dependsOn` | Structural differences |
| Model lineage | `pedigree.ancestors` | `ancestorOf` | |
| Training datasets | `modelCard.datasets` | `trainedOn` | |
| Test datasets | `modelCard.datasets` in `formulation` | `testedOn` | Role distinction differs |

---

## Notes on Mapping Philosophy

- This document focuses on **semantic alignment**, not serialization mechanics.
- Where a format lacks a native construct, **extensions or partial mappings** are noted explicitly.
- Structural differences (tree vs graph) are handled by format-specific tooling, not by redefining fields.

---

## Implementation Notes (Non-Normative)

The AIBOM Generator uses:
- a **format-agnostic internal data model**
- a **machine-readable mapping registry** to project data into each format specification
- separate emitters for CycloneDX and SPDX

Detailed implementation specs are intentionally **out of scope** for this document and do not affect the validity of the field mappings defined here.

## G7 SBOM for AI Minimum Elements

This **draft project crosswalk** maps the G7 guidance's 50 elements to
[CycloneDX 1.7](https://cyclonedx.org/docs/1.7/json/). It has not been approved by
G7, BSI, or the CycloneDX working group. The machine-readable mapping lives in
[`g7_cyclonedx_1_7_profile.json`](../../src/models/g7_cyclonedx_1_7_profile.json).
Each row contains the original element name, scope, printed source page and
section, candidate CycloneDX fields, executable evidence groups, and limitations.

### Verified source

Element names, clusters and definitions were checked on 18 October 2026 against
the [original BSI-hosted publication](https://www.bsi.bund.de/SharedDocs/Downloads/EN/BSI/KI/SBOM-for-AI_minimum-elements.pdf?__blob=publicationFile&v=4),
*Software Bill of Materials for AI — Minimum Elements*, sections 2.1–2.7,
printed pages 5–22. Its SHA-256 is
`d0b2f41c5baa2bcb2c8bea5b9d610fe9bafef202bbc42798b0a562eb0db5e77c`.
The PDF's first two pages precede printed page 1. The source URL's `v=4` is a
download parameter, not an asserted edition number. The digest identifies the
exact reviewed artifact. The original guidance is advisory and does not prescribe
the scoring algorithm below.

| Cluster | Elements |
| --- | ---: |
| Metadata (§2.1) | 10 |
| System Level Properties (§2.2) | 9 |
| Models (§2.3) | 13 |
| Datasets Properties (§2.4) | 10 |
| Infrastructure (§2.5) | 2 |
| Security Properties (§2.6) | 4 |
| Key Performance Indicators (§2.7) | 2 |

### Using the profile

```python
from src.models.g7_profile import calculate_g7_completeness

report = calculate_g7_completeness(cyclonedx_document)
print(report["status"], report["percentage"])
for element in report["elements"]:
    print(element["id"], element["status"], element["subjects"])
```

The existing `calculate_completeness_score()` response includes this report under
`g7_profile`. Existing Basic/Standard/Advanced scores are unchanged. The profile
requires a schema-valid CycloneDX 1.7 JSON document and uses the bundled library
schema; it never downloads schemas, model files, or external evidence.
CycloneDX 1.6 is reported as `unsupported`, malformed documents as `invalid`, and
documents without typed AI models or datasets as `not_applicable`. Those reports
have `percentage: null` and all 50 rows marked `not_assessed`.

| Element status | Meaning |
| --- | --- |
| `present` | All selected subjects have the explicitly checked native evidence. |
| `partial` | Some subjects or required evidence groups are missing, or native fields cover only part of the meaning. |
| `missing` | The selected subjects lack known evidence for an automatic check. |
| `unknown` | The field explicitly declares unknown information. |
| `review` | Human judgment is required, a dataset reference is unresolved, or the applicable subject is not identified. |
| `not_assessed` | The document is unsupported, invalid, or has no typed AI subjects. |

The percentage is `100 × present / 50`, a **project-defined known-evidence
coverage measure**. Partial, unknown and review rows receive no full credit;
they remain in the fixed denominator. The guidance explicitly allows unknown
values for several elements, so this score must not be used as a conformance
pass/fail threshold. Manual rows cannot earn automatic credit, even in a richly
documented BOM. `present` does not establish correctness, completeness of the
real-world inventory, artifact integrity, licensing rights or legal compliance.

### Evidence scope and review boundaries

- Model checks select every `machine-learning-model` component under
  `metadata.component`, `components[]`, and nested component assemblies.
- Dataset checks select `data` components containing dataset records, inline
  `modelCard.modelParameters.datasets[]`, and referenced data components.
  Repeated references to the same component are assessed once. Unresolved and
  external references remain visible for review; they are never fetched.
- A system is selected only when the application in `metadata.component`
  contains or depends on a typed model. A model BOM alone does not describe its
  deployment system. Other system representations currently need review.
- Evidence groups are ANDed; paths within each group are alternatives. Paths
  are relative to the selected subject, `*` traverses arrays only, and reported
  evidence uses JSON pointers into the input. Every selected subject must pass
  before an element can be `present`. Empty values and unknown placeholders
  cannot satisfy an evidence group.
- Dependency checks resolve source/target components. A model-to-dataset edge
  does not establish runtime software or dataset creation-tool dependencies.
  Component BOM references must be unique. Graph completeness and derivation
  semantics still require review.
- Signatures require author attribution and cryptographic verification; hash
  algorithm suitability needs review. Hash values are not recomputed.
- The producer is a manufacturer, not necessarily a supplier. The BOM author is
  the generation-tool operator, not automatically the component producer.
- **Model license** checks license-type documentation. The source's openness
  examples are illustrative and not a separately mandatory 51st element; this
  profile makes no claim about weights, architecture, data or training openness.
- **Dataset content/provenance** use `data[].contents` and `data[].governance`.
  These are partial evidence; content descriptions, collection and processing
  methods still require interpretation.
- **Cybersecurity policy information** concerns the producer's `security.txt`.
  A matching URL is partial evidence; an email address is insufficient.
  **Operational performance KPIs** concern uptime, incident response, latency,
  throughput and load balancing. Generic model accuracy is insufficient.

No custom `genai:aibom:g7:*` properties are generated or registered. Arbitrary
properties never bypass native checks or turn review findings into passes.
Manual rows document candidate locations for a future reviewed extension or
attestation mapping; accepting such evidence is deliberately outside this
initial automated profile.

---

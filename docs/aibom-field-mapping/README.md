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
| Training information | Partial (modelCard) | `informationAboutTraining` | |
| Data preprocessing | Extension | `modelDataPreprocessing` | |
| Sensitive data usage | Partial | `useSensitivePersonalInformation` | |
| Explainability information | Extension | `modelExplainability` | |
| Known limitations | `technicalLimitations` | `limitation` | |
| Safety risk assessment | Partial | `safetyRiskAssessment` | |
| Runtime dependencies | `dependencies` | `dependsOn` | Structural differences |
| Model lineage | `pedigree.ancestors` | `ancestorOf` | |
| Training datasets | `modelCard.datasets` | `trainedOn` | |
| Test datasets | `modelCard.datasets` | `testedOn` | Role distinction differs |

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

---

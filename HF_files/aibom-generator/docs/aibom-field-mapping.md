# AIBOM Canonical Field & Relationship Mapping

## Scope and Purpose

This document defines the **canonical field and relationship model** used by the AIBOM Generator and how it maps to:

- **CycloneDX 1.6**
- **SPDX 3.0 (AI Profile)**

The goal of this document is to:
- provide a **single source of truth** for what data the tool collects,
- show how each concept maps to both standards,
- enable **clean, maintainable implementation**,
- allow **external reviewers (e.g., SPDX experts)** to validate mapping choices independently of code.

This document is **normative for data modeling**, not for code structure or scoring weights.

---

## How to Read This Document

- **Table 0** defines the **canonical concepts**.  
  This is the authoritative list of fields and relationships the tool understands.

- **Table 1** shows **conceptual coverage** across standards.  
  This is the most important table for reviewers and alignment discussions.

- **Table 2** shows **CycloneDX 1.6 projection details** (tree-based, JSON-path oriented).

- **Table 3** shows **SPDX 3.0 projection details** (graph-based, object + relationship oriented).

Scoring, validation, and emitters operate on the **canonical model**, not directly on these tables.

---

## Standards Covered

- **CycloneDX:** v1.6  
- **SPDX:** v3.0 (AI Profile)

Future versions are expected to be supported by **updating mappings**, not by redefining canonical concepts.

---

## Table of Contents

- [Table 0 — Canonical Concepts Taxonomy](#table-0--canonical-concepts-taxonomy)
- [Table 1 — Consolidated Coverage Map](#table-1--consolidated-coverage-map)
- [Table 2 — CycloneDX 1.6 Projection Specification](#table-2--cyclonedx-16-projection-specification)
- [Table 3 — SPDX 3.0 Projection Specification](#table-3--spdx-30-projection-specification)
- [Technical Implementation Notes](#technical-implementation-notes)

---

## Table 0 — Canonical Concepts Taxonomy

Defines **what the tool models**, independent of any standard.

| Canonical concept | Concept type | Category | Tier | Primary object | Notes |
|---|---|---|---|---|---|
| bom_format | attribute | identity_provenance | critical | BOM | Baseline validity |
| bom_spec_version | attribute | identity_provenance | critical | BOM | Baseline validity |
| bom_serial_number | attribute | identity_provenance | critical | BOM | Baseline validity |
| bom_version | attribute | identity_provenance | critical | BOM | Baseline validity |
| model_type | attribute | identity_provenance | important | Model | Model vs system |
| model_name | attribute | identity_provenance | critical | Model | Stable identifier |
| model_version_string | attribute | identity_provenance | critical | Model | Version |
| model_description | attribute | characteristics_limitations | supplementary | Model | Narrative |
| model_purl | attribute | identity_provenance | critical | Model | Cross-ecosystem ID |
| model_download_url | attribute | identity_provenance | critical | Model | Distribution |
| model_vcs_url | attribute | identity_provenance | supplementary | Model | Source repo |
| supplied_by | attribute | identity_provenance | critical | Model | Supplier |
| licenses | attribute + relationship | identity_provenance | important | Model | Licensing |
| primary_purpose | attribute | operational_technical_metadata | important | Model | Task / intent |
| domain | attribute | operational_technical_metadata | supplementary | Model | NLP, CV, etc. |
| autonomy_type | attribute | operational_technical_metadata | supplementary | Model | HITL |
| standard_compliance | attribute | identity_provenance | supplementary | Model | Claimed compliance |
| type_of_model | attribute | operational_technical_metadata | important | Model | Learning type |
| hyperparameters | composite | operational_technical_metadata | important | Model | Key–value |
| performance_metrics | composite | operational_technical_metadata | supplementary | Model | Metrics |
| metric_decision_thresholds | composite | operational_technical_metadata | supplementary | Model | Thresholds |
| information_about_training | attribute | dependencies_lineage | important | Model | Training narrative |
| model_data_preprocessing | composite | dependencies_lineage | supplementary | Model | Preprocessing |
| model_explainability | composite | characteristics_limitations | supplementary | Model | Explainability |
| limitations | attribute | characteristics_limitations | important | Model | Known limits |
| safety_risk_assessment | attribute | characteristics_limitations | important | Model | Risk |
| use_sensitive_personal_information | attribute | dependencies_lineage | supplementary | Model | Presence |
| energy_consumption | composite | operational_technical_metadata | supplementary | Model | Energy |
| runtime_dependencies | relationship | dependencies_lineage | critical | Model → Component | Runtime deps |
| model_lineage | relationship | dependencies_lineage | important | Model → Model | Lineage |
| training_datasets | relationship | dependencies_lineage | critical | Model → Dataset | Trained on |
| test_datasets | relationship | dependencies_lineage | supplementary | Model → Dataset | Tested on |

[⬆ Back to top](#aibom-canonical-field--relationship-mapping)

---

## Table 1 — Consolidated Coverage Map

Conceptual alignment across both standards.

| Canonical concept | CycloneDX 1.6 | SPDX 3.0 | Support type | Notes |
|---|---|---|---|---|
| model_name | component.name | AIPackage.name | native / native | 1:1 |
| model_version_string | component.version | AIPackage.packageVersion | native / native |  |
| model_description | component.description | summary / description | native / native | Optional |
| model_purl | component.purl | packageUrl | native / native |  |
| model_download_url | extRef(distribution) | downloadLocation | native / native | SPDX stricter |
| model_vcs_url | extRef(vcs) | ExternalRef | native / relationship |  |
| supplied_by | component.supplier | suppliedBy | native / native | Required |
| licenses | component.licenses | License relationships | native / relationship |  |
| primary_purpose | modelCard.task | primaryPurpose | native / native | Required |
| domain | tags / narrative | domain | native / native |  |
| autonomy_type | extension | autonomyType | extension / native |  |
| standard_compliance | attestations | standardCompliance | relationship / native |  |
| type_of_model | modelCard | typeOfModel | native / native |  |
| hyperparameters | modelCard | hyperparameter | native / native |  |
| performance_metrics | modelCard | metric | native / native |  |
| metric_decision_thresholds | extension | metricDecisionThreshold | extension / native |  |
| information_about_training | datasets + text | informationAboutTraining | native / native |  |
| model_data_preprocessing | extension | modelDataPreprocessing | extension / native |  |
| model_explainability | extension | modelExplainability | extension / native |  |
| limitations | modelCard | limitation | native / native |  |
| safety_risk_assessment | modelCard | safetyRiskAssessment | native / native |  |
| use_sensitive_personal_information | dataset info | useSensitivePersonalInformation | native / native |  |
| energy_consumption | modelCard | energyConsumption | native / native |  |
| runtime_dependencies | dependencies graph | dependsOn | relationship / relationship |  |
| model_lineage | pedigree | ancestorOf | relationship / relationship | Policy |
| training_datasets | datasets | trainedOn | relationship / relationship | Policy |
| test_datasets | tagged datasets | testedOn | relationship / relationship | Policy |

[⬆ Back to top](#aibom-canonical-field--relationship-mapping)

---

## Table 2 — CycloneDX 1.6 Projection Specification

Technical placement for CycloneDX emitters.

| Canonical concept | CDX object | Location | Cardinality |
|---|---|---|---|
| model_name | component | components[type=machine-learning-model].name | 1 |
| model_version_string | component | .version | 0..1 |
| model_description | component | .description | 0..1 |
| model_purl | component | .purl | 0..1 |
| supplied_by | component | .supplier | 0..1 |
| licenses | component | .licenses | 0..n |
| model_download_url | component | externalReferences[type=distribution] | 0..n |
| model_vcs_url | component | externalReferences[type=vcs] | 0..n |
| primary_purpose | modelCard | modelParameters.task | 0..1 |
| type_of_model | modelCard | modelParameters.approach | 0..n |
| hyperparameters | modelCard | modelParameters | 0..n |
| performance_metrics | modelCard | quantitativeAnalysis | 0..n |
| limitations | modelCard | technicalLimitations | 0..n |
| energy_consumption | modelCard | environmentalConsiderations | 0..1 |
| runtime_dependencies | BOM | dependencies[] | 0..n |
| model_lineage | component | pedigree.ancestors | 0..n |
| training_datasets | dataset | modelParameters.datasets | 0..n |
| test_datasets | dataset | datasets + role tag | 0..n |

[⬆ Back to top](#aibom-canonical-field--relationship-mapping)

---

## Table 3 — SPDX 3.0 Projection Specification

Graph-based mapping using a **minimal deterministic policy**.

| Canonical concept | SPDX object | Property | Relationship | Cardinality |
|---|---|---|---|---|
| model_name | AIPackage | name | — | 1 |
| model_version_string | AIPackage | packageVersion | — | 1 |
| model_description | AIPackage | summary / description | — | 0..1 |
| model_purl | AIPackage | packageUrl | — | 0..1 |
| model_download_url | AIPackage | downloadLocation | — | 1 |
| supplied_by | AIPackage | suppliedBy | — | 1 |
| primary_purpose | AIPackage | primaryPurpose | — | 1 |
| domain | AIPackage | domain | — | 0..* |
| autonomy_type | AIPackage | autonomyType | — | 0..1 |
| type_of_model | AIPackage | typeOfModel | — | 0..* |
| hyperparameters | AIPackage | hyperparameter | — | 0..* |
| performance_metrics | AIPackage | metric | — | 0..* |
| metric_decision_thresholds | AIPackage | metricDecisionThreshold | — | 0..* |
| information_about_training | AIPackage | informationAboutTraining | — | 0..1 |
| model_data_preprocessing | AIPackage | modelDataPreprocessing | — | 0..* |
| model_explainability | AIPackage | modelExplainability | — | 0..* |
| limitations | AIPackage | limitation | — | 0..1 |
| safety_risk_assessment | AIPackage | safetyRiskAssessment | — | 0..1 |
| standard_compliance | AIPackage | standardCompliance | — | 0..* |
| use_sensitive_personal_information | AIPackage | useSensitivePersonalInformation | — | 0..1 |
| energy_consumption | AIPackage | energyConsumption | — | 0..1 |
| runtime_dependencies | AIPackage → Artifact | — | dependsOn | 0..* |
| model_lineage | AIPackage → AIPackage | — | ancestorOf | 0..* |
| training_datasets | AIPackage → Dataset | — | trainedOn | 0..* |
| test_datasets | AIPackage → Dataset | — | testedOn | 0..* |
| licenses | AIPackage → License | — | hasDeclaredLicense / hasConcludedLicense | 1 each |

[⬆ Back to top](#aibom-canonical-field--relationship-mapping)

---

## Technical Implementation Notes (High Level)

- Extractors normalize raw source metadata into a **canonical AIBOM object**.
- The canonical AIBOM object is the **single authoritative internal representation** of all extracted fields and relationships.
- Both **scoring** and **format emitters** consume the canonical AIBOM object independently.
- Scoring evaluates **canonical field and relationship coverage**, independent of any output format.
- Emitters project canonical data into format-specific representations (CycloneDX, SPDX, etc.) using mapping and policy tables.
- Format compliance checks (e.g., required SPDX AI Profile fields) are applied **only during emission**, not during scoring.
- This separation allows standards to evolve without impacting extractors or scoring logic.



---

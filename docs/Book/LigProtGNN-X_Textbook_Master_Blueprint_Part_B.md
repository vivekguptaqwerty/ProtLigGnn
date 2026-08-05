# LigProtGNN-X Textbook Master Blueprint (BGS v1.0)

# Part B — Repository Analysis & Knowledge Extraction Specification

> **Objective:** Define how the entire LigProtGNN-X repository shall be reverse engineered before a single textbook chapter is written.

---

# 1. Purpose

The textbook must accurately represent the implementation. Therefore, every chapter shall be grounded in a systematic analysis of the complete codebase and associated assets.

No implementation detail may be invented or inferred without evidence from the repository or cited literature.

---

# 2. Repository Analysis Goals

The analysis process shall reconstruct:

- Scientific objectives
- Biological workflow
- Mathematical foundations
- Model architecture
- Data pipeline
- Training pipeline
- Evaluation pipeline
- Reliability framework
- Explainability framework
- Robustness framework
- Engineering architecture
- Deployment workflow
- Historical evolution across project phases

The result is a structured knowledge graph that maps repository artifacts to textbook chapters.

---

# 3. Scope of Analysis

Every directory and file shall be considered unless explicitly identified as generated build artifacts.

The analysis shall inspect, where present:

- README files
- Markdown documentation
- PDFs and reports
- Research notes
- Python modules
- Jupyter notebooks
- Configuration files (YAML, JSON, TOML, INI)
- Dataset manifests
- CSV and TSV data
- Training logs
- Evaluation logs
- Checkpoints metadata
- Images and diagrams
- Visualization scripts
- Test suites
- CI/CD configuration
- Docker and environment files
- Requirements and dependency manifests
- Licensing information

---

# 4. Multi-Pass Analysis Workflow

## Pass 1 — Structural Inventory

Create a complete inventory of:

- Directories
- Packages
- Modules
- Scripts
- Assets
- Reports

Generate a repository tree.

---

## Pass 2 — Dependency Mapping

Identify:

- Internal module dependencies
- External libraries
- Data dependencies
- Configuration dependencies
- Runtime interactions

Produce dependency graphs.

---

## Pass 3 — Functional Classification

Classify every significant component into categories such as:

- Data ingestion
- Preprocessing
- Protein encoding
- Ligand encoding
- Fusion
- Prediction
- Training
- Evaluation
- Visualization
- Explainability
- Utilities
- Testing

---

## Pass 4 — Scientific Interpretation

For each major module determine:

- Biological motivation
- Mathematical basis
- Machine learning concepts
- Engineering purpose
- Inputs
- Outputs
- Assumptions
- Limitations

---

## Pass 5 — Historical Evolution

Identify the progression of the project.

Capture:

- Major milestones
- Architectural revisions
- Performance improvements
- Rejected approaches
- Lessons learned

Map these to the documented development phases.

---

# 5. Repository-to-Book Mapping

Every important repository component shall be mapped to:

- Part
- Chapter
- Section
- Figures
- Equations
- Algorithms
- Tables
- Case studies

No major implementation component should remain undocumented.

---

# 6. Knowledge Extraction Rules

For every important class, function, notebook, or report capture:

- Name
- Purpose
- Inputs
- Outputs
- Dependencies
- Related datasets
- Related experiments
- Related configurations
- Mathematical concepts
- Biological concepts
- Engineering decisions
- Related textbook chapters

---

# 7. Documentation Evidence Policy

Every implementation claim should be traceable to one or more repository artifacts.

Each extracted fact should retain provenance such as:

- Source file
- Module
- Class
- Function
- Report
- Configuration
- Experiment

This provenance enables future verification and updates.

---

# 8. Knowledge Graph

Construct an internal conceptual graph linking:

Repository Artifact
→ Scientific Concept
→ Mathematical Concept
→ Engineering Concept
→ Experimental Evidence
→ Textbook Chapter

This graph becomes the foundation for chapter generation.

---

# 9. Phase Mapping

The repository analysis shall explicitly document the evolution through all project phases.

For every phase record:

- Objectives
- New capabilities
- Architectural changes
- Experimental outcomes
- Reliability improvements
- Remaining limitations
- Links to corresponding chapters

---

# 10. Quality Assurance

Before chapter generation begins verify that:

- Every significant directory has been inspected.
- Every major module has a documented purpose.
- Every experiment has context.
- Every report has been reviewed.
- Repository coverage is effectively complete.
- Outstanding ambiguities are recorded instead of guessed.

---

# 11. Deliverables of Repository Analysis

Produce and maintain:

- Repository inventory
- Dependency map
- Component catalogue
- Dataset catalogue
- Configuration catalogue
- Experiment catalogue
- Knowledge graph
- Repository-to-book mapping matrix
- Phase evolution summary

These artifacts become mandatory references for all subsequent documentation work.

---

# End of Part B

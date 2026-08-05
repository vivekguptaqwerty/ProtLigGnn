"""Generate B.E. Final Year Project review presentation for ProtLigGNN."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "presentation" / "ProtLigGNN_FYP_Review.pptx"

# Verified metrics from repository artifacts (not fabricated)
METRICS = {
    "best_run": "no_crossgraph_5000_e40",
    "best_epoch": 16,
    "pcc": 0.6714,
    "spearman": 0.6624,
    "rmse": 1.3722,
    "mae": 1.0677,
    "casf_pearson": 0.371,
    "casf_pearson_ci": "[0.275, 0.461]",
    "casf_spearman": 0.385,
    "casf_rmse": 2.111,
    "casf_mae": 1.705,
    "casf_r2": 0.054,
    "casf_top1": 35.09,
    "casf_top2": 52.63,
    "casf_top3": 73.68,
    "casf_preprocessed": "285/285",
    "casf_time_s": 21.02,
    "casf_fallback_pct": 32.28,
    "pdbbind_total": "19,443",
    "train_samples": 5000,
    "gen_generated": 10000,
    "gen_valid": 6056,
    "gen_valid_pct": 60.56,
    "gen_unique": 4151,
    "gen_novel": 3894,
    "gen_novel_pct": 93.81,
    "best_vina": -7.8,
    "unit_tests": 111,
    "tests_pass": "100%",
}

SLIDES = [
    {
        "title": "ProtLigGNN",
        "subtitle": "Graph Neural Network for Protein–Ligand Binding Affinity Prediction\nwith Generative Chemistry Closed Loop",
        "bullets": [
            "Bachelor of Engineering — Final Year Project Review",
            "Candidate: [Your Name] | Department: [Your Department]",
            "Guide: [Guide Name] | Institution: [Institution Name]",
            "August 2026",
        ],
        "layout": "title",
    },
    {
        "title": "Presentation Agenda",
        "bullets": [
            "1. Introduction & Problem Statement",
            "2. Data Engineering (emphasis): datasets, preprocessing, feature dimensions",
            "3. Graph construction & dimensionality flow",
            "4. GNN algorithms & training pipeline",
            "5. Experimental results (PDBbind, CASF-2016, GenAI, docking)",
            "6. Limitations, future work & conclusion",
        ],
    },
    {
        "title": "Introduction — Drug Discovery Context",
        "bullets": [
            "Protein–ligand binding affinity (pKd) quantifies how strongly a drug candidate binds its target.",
            "Traditional high-throughput screening explores only a tiny fraction of chemical space (~10⁶⁰ drug-like molecules).",
            "Computational methods can score candidate complexes in milliseconds from 3D structure.",
            "ProtLigGNN predicts binding affinity from paired molecular graphs (ligand + protein pocket).",
            "Extended with a closed loop: SMILES generation → RDKit filtering → GNN scoring → GAPS ranking → Vina docking.",
        ],
    },
    {
        "title": "Problem Statement",
        "bullets": [
            "Physics-based scoring functions struggle with conformational flexibility and entropy.",
            "Sequence-only or 2D graph models ignore 3D spatial geometry of binding sites.",
            "Generative models produce many invalid molecules without downstream filtering.",
            "Need: a reproducible pipeline from raw PDBbind structures → graph features → affinity prediction → candidate prioritization.",
        ],
    },
    {
        "title": "Project Objectives",
        "bullets": [
            "Build an end-to-end data pipeline from PDBbind v2020 complexes to PyG graph pairs.",
            "Engineer fixed-dimension node features: 78-D (ligand atoms) + 30-D (protein residues).",
            "Train a dual-encoder GNN (GAT ligand + GCN protein) for pKd regression.",
            "Validate externally on CASF-2016 CoreSet (285 held-out complexes, 57 ranking groups).",
            "Integrate GenAI ligand generation with affinity-guided GAPS ranking and AutoDock Vina validation.",
        ],
    },
    {
        "title": "Technology Stack",
        "bullets": [
            "Python 3 | PyTorch 2.3 | PyTorch Geometric (GATConv, GCNConv, Batch)",
            "RDKit — ligand parsing, SMILES validity, QED/Lipinski descriptors",
            "BioPython — PDB parsing, amino-acid pocket extraction",
            "NumPy, SciPy, scikit-learn, pandas, matplotlib",
            "AutoDock Vina + Open Babel — secondary docking validation",
            "pytest — 111 unit/integration tests (100% pass rate, FINAL_VALIDATION_REPORT.md)",
        ],
    },
    {
        "title": "DATA — Dataset Overview",
        "section": "Data Engineering",
        "bullets": [
            f"PDBbind v2020 (general set): {METRICS['pdbbind_total']} protein–ligand complexes with measured binding constants.",
            f"Training experiments: {METRICS['train_samples']:,} complexes sampled from PDBbind (max_samples flag).",
            "Split: 80% train / 10% validation / 10% test (random, seed-controlled).",
            "CASF-2016 CoreSet: 285 complexes in 57 target groups — strict external benchmark (never seen in training).",
            "Target variable: pKd = −log₁₀(Kd in molar units), parsed from INDEX_general_PL.2020R1.lst.",
        ],
    },
    {
        "title": "DATA — Raw Input Structure",
        "section": "Data Engineering",
        "bullets": [
            "Per complex: {pdb_id}_protein.pdb + {pdb_id}_ligand.sdf (or .mol2 fallback).",
            "Index file maps PDB ID → binding constant (Kd, Ki, IC50) → converted to pKd.",
            "Discovery: recursive scan of pdbbind_subset/{year}/{pdb_id}/ directories.",
            "Ligand loader: SDF with sanitize=True; fallback unsanitized SDF or MOL2 on valency errors.",
            "Requirement: ligand must have a 3D conformer (GetNumConformers > 0) — complexes without 3D coords are skipped.",
        ],
    },
    {
        "title": "DATA — Target Label Pipeline",
        "section": "Data Engineering",
        "bullets": [
            "Affinity index regex extracts numeric binding constants from PDBbind INDEX file.",
            "Units normalized to molar; pKd = −log₁₀(Kd_M).",
            "Optional z-score target normalization (--target_normalization) from train-set statistics.",
            "Default: raw pKd regression with MSE loss (not normalized).",
            "Typical pKd range in dataset: ~2 to 12 (nanomolar to femtomolar binders).",
        ],
    },
    {
        "title": "DATA — Ligand Preprocessing",
        "section": "Data Engineering",
        "bullets": [
            "Input: SDF/MOL2 with 3D coordinates → RDKit Mol object.",
            "Graph nodes = atoms; undirected bonds → bidirectional edges in edge_index.",
            "3D positions stored separately in pos tensor (N_atoms × 3) for geometry modules.",
            "CASF preprocessing audit: 285/285 complexes processed in 21.02 s (100% success).",
            "32.28% required MOL2 fallback due to SDF sanitization/valency errors.",
        ],
    },
    {
        "title": "DATA — Ligand Feature Vector (78-D)",
        "section": "Data Engineering",
        "bullets": [
            "Atomic number one-hot (10 classes + unknown): C, N, O, S, halogens, P, B, etc.",
            "Degree, formal charge, hybridization, H-count, implicit/total valence one-hots.",
            "Chirality tag, ring membership (3–8 member rings), aromaticity flags.",
            "Normalized: mass, VdW radius, covalent radius, isotope, radical electrons.",
            "Pharmacophore flags: H-bond donor/acceptor, acidic/basic, halogen, metal.",
            "Fixed LIGAND_FEATURE_DIM = 78 (enforced at runtime in protliggnn_train.py).",
        ],
    },
    {
        "title": "DATA — Protein Pocket Extraction",
        "section": "Data Engineering",
        "bullets": [
            "Parse PDB with BioPython PDBParser; standard amino acids only (is_aa).",
            "Pocket definition: residues with any heavy atom within 6.0 Å of any ligand atom.",
            "Residue center: Cα coordinate, or centroid of heavy atoms if Cα missing.",
            "Hydrogens excluded from distance calculations for pocket selection.",
            "Typical pocket size: 20–80 residues depending on ligand size and binding site depth.",
        ],
    },
    {
        "title": "DATA — Protein Feature Vector (30-D)",
        "section": "Data Engineering",
        "bullets": [
            "20-D amino-acid one-hot (ALA, ARG, … VAL).",
            "5-D physicochemical group flags: aromatic, hydrophobic, polar, positive, negative.",
            "Special residue flags: GLY, PRO, Cα presence.",
            "Normalized residue mass (÷250) and heavy-atom count proxy (÷10).",
            "Fixed PROTEIN_FEATURE_DIM = 30 (enforced at runtime).",
            "Edge construction: connect residues with centers within 8.0 Å (sequential fallback if isolated).",
        ],
    },
    {
        "title": "DATA — Graph Topology & Dimensionality Flow",
        "section": "Data Engineering",
        "bullets": [
            "Ligand graph: (N_lig × 78) → GAT encoder → (N_lig × 256).",
            "Protein graph: (N_prot × 30) → GCN encoder → (N_prot × 256).",
            "Global mean pool → ligand vector (256) + protein vector (256) = joint (512).",
            "MLP regressor: 512 → 512 → 128 → 1 (Dropout 0.2/0.1).",
            "No PCA/StandardScaler on input features during training — dimensionality is fixed by design.",
            "Embedding analysis (post-hoc): PCA on GNN embeddings shows PC2 vs affinity R = −0.31.",
        ],
    },
    {
        "title": "DATA — Quality, Splits & Reproducibility",
        "section": "Data Engineering",
        "bullets": [
            "Train/val/test: 80/10/10 random split with configurable seed.",
            "Skipped complexes logged with reason (missing conformer, parse failure, no affinity label).",
            "LigProtGNN-X framework: Pydantic-validated dataset registry + frozen v4.0 config.",
            "Graph caching & mixed-precision (v1.1): ~66% epoch time reduction for benchmarking.",
            "111 pytest tests across preprocessing, models, uncertainty, explainability modules.",
        ],
    },
    {
        "title": "ALGORITHM — ProtLigGNN Architecture",
        "section": "Model & Training",
        "bullets": [
            "Dual heterogeneous graphs: ligand (atom-level) + protein pocket (residue-level).",
            "Ligand encoder: 3× GATConv layers (4 heads → 4 heads → 1 head), hidden_dim=256.",
            "Protein encoder: 3× GCNConv layers, hidden_dim=256, ReLU + LayerNorm.",
            "Optional cross-graph bidirectional MultiheadAttention (4 heads) — ablated in best run.",
            "Output: scalar pKd prediction via MSE regression.",
        ],
    },
    {
        "title": "ALGORITHM — Encoder & Fusion Details",
        "section": "Model & Training",
        "bullets": [
            "GAT: attention-weighted message passing captures important atomic interactions.",
            "GCN: efficient convolution over residue adjacency within binding pocket.",
            "Cross-attention (when enabled): ligand atoms attend to protein residues and vice versa.",
            "Best run used --no_crossgraph: separate encoders + concat pooled embeddings.",
            "Extended variants (LigProtGNN-X): geometry attention (RBF bias), EGNN, ESM-2+ChemBERTa hybrid.",
        ],
    },
    {
        "title": "ALGORITHM — Training Configuration",
        "section": "Model & Training",
        "bullets": [
            "Optimizer: Adam (lr=1e-3, weight decay=1e-5); batch size=8.",
            "Epochs: 40–50; early stopping patience=10 on validation RMSE.",
            "Loss: MSE on pKd; metrics: PCC, Spearman, RMSE, MAE.",
            "Best checkpoint saved by lowest val RMSE; restored before test evaluation.",
            "Device: CPU/GPU; outputs: checkpoint .pt, epoch CSV, test predictions CSV, scatter plots.",
        ],
    },
    {
        "title": "RESULTS — PDBbind Internal Test (Best Run)",
        "section": "Experimental Results",
        "bullets": [
            f"Run: {METRICS['best_run']} | Best epoch: {METRICS['best_epoch']} | Samples: {METRICS['train_samples']:,}",
            f"Pearson Correlation (PCC): {METRICS['pcc']:.4f}",
            f"Spearman: {METRICS['spearman']:.4f}",
            f"RMSE: {METRICS['rmse']:.4f} pKd units",
            f"MAE: {METRICS['mae']:.4f} pKd units",
            "Source: outputs/final_research_package/key_metrics_table.md",
        ],
    },
    {
        "title": "RESULTS — Architecture Ablation (5,000 samples, 40 epochs)",
        "section": "Experimental Results",
        "bullets": [
            "no_crossgraph_5000_e40 — PCC 0.6714 | Spearman 0.6624 | RMSE 1.3722 | MAE 1.0677",
            "crossgraph_5000_e40 — PCC 0.6570 | Spearman 0.6495 | RMSE 1.3853 | MAE 1.0642",
            "crossgraph_attention_5000_e40 — PCC 0.6023 | Spearman 0.5777 | RMSE 1.4905 | MAE 1.1832",
            "Finding: simpler no-crossgraph variant generalizes best on this split.",
            "Sample-size ablation (500 vs 2000): PCC improves from ~0.34 to ~0.51 with more data.",
        ],
    },
    {
        "title": "RESULTS — CASF-2016 External Benchmark",
        "section": "Experimental Results",
        "bullets": [
            f"Coverage: {METRICS['casf_preprocessed']} complexes, 57 ranking groups.",
            f"Scoring — Pearson R: {METRICS['casf_pearson']:.3f} (95% CI {METRICS['casf_pearson_ci']})",
            f"Scoring — Spearman: {METRICS['casf_spearman']:.3f} | RMSE: {METRICS['casf_rmse']:.3f} | MAE: {METRICS['casf_mae']:.3f}",
            f"Scoring — R²: {METRICS['casf_r2']:.3f} (expected generalization gap vs internal test)",
            f"Ranking — Top-1: {METRICS['casf_top1']:.2f}% | Top-2: {METRICS['casf_top2']:.2f}% | Top-3: {METRICS['casf_top3']:.2f}%",
            "Source: reports/CASF_Evaluation/14_Final_Benchmark_Report.md",
        ],
    },
    {
        "title": "RESULTS — GenAI Closed Loop",
        "section": "Experimental Results",
        "bullets": [
            "Pipeline: PDBbind SMILES → char-level LSTM → RDKit filter → ProtLigGNN score → GAPS rank → Vina.",
            "GAPS = Generative Affinity Prioritization Score (affinity + novelty + QED + diversity).",
            f"Generated: {METRICS['gen_generated']:,} molecules",
            f"RDKit-valid: {METRICS['gen_valid']:,} ({METRICS['gen_valid_pct']:.2f}%)",
            f"Unique valid: {METRICS['gen_unique']:,} | Novel unique: {METRICS['gen_novel']:,} ({METRICS['gen_novel_pct']:.2f}% novelty)",
            "Novelty = not present in PDBbind training ligand set.",
        ],
    },
    {
        "title": "RESULTS — Docking Validation & Top Candidates",
        "section": "Experimental Results",
        "bullets": [
            f"Best generated Vina score: {METRICS['best_vina']:.1f} kcal/mol",
            "Best SMILES: NS(=O)(=O)c1cccc(-c2cc(C(F)(F)F)ccc2O)c1",
            "ProtLigGNN predicted pKd: 7.37 | GAPS: 0.79 | QED: 0.89 | Lipinski violations: 0",
            "Known binder example: PDB 5kqy — predicted 9.25 vs true 9.29 (|error| = 0.04)",
            "Top generated candidates: Vina scores −6.2 to −7.8 kcal/mol (secondary validation only).",
        ],
    },
    {
        "title": "RESULTS — Software Validation",
        "section": "Experimental Results",
        "bullets": [
            f"Unit/integration tests: {METRICS['unit_tests']} tests — {METRICS['tests_pass']} pass rate",
            "Phases certified: dataset registry, protein parser, geometry attention, uncertainty, robustness.",
            "LigProtGNN-X v4.0 architecture frozen (ARCHITECTURE_FREEZE.md, July 2026).",
            "Multi-seed benchmark (Phase 5.5): val Pearson 0.834–0.852 across 5 seeds.",
            "All metrics sourced from logged CSV/JSON artifacts — reproducible via protliggnn_train.py.",
        ],
    },
    {
        "title": "Limitations & Future Work",
        "bullets": [
            "No wet-lab experimental binding validation — all results are computational.",
            "CASF external Pearson (~0.37) << internal PDBbind PCC (~0.67) — domain shift expected.",
            "SMILES LSTM is lightweight; chemical quality depends on RDKit post-filtering.",
            "Future: transformer/graph generators, protein-conditioned generation, MD validation.",
            "Future: active learning loop from GAPS-ranked candidates.",
        ],
    },
    {
        "title": "Conclusion",
        "bullets": [
            "Built a complete data-to-prediction pipeline: 78-D + 30-D features → dual GNN → pKd.",
            "Best internal test: PCC 0.6714, RMSE 1.3722 on 5,000 PDBbind complexes.",
            "External CASF-2016 validation confirms ranking capability (73.68% top-3 success).",
            "Closed-loop GenAI integration: 60.56% validity, 93.81% novelty, Vina −7.8 kcal/mol best.",
            "Thank you — questions welcome.",
        ],
        "layout": "title",
    },
]

ACCENT = RGBColor(0x1A, 0x56, 0x7E)
DARK = RGBColor(0x2C, 0x3E, 0x50)
MUTED = RGBColor(0x5D, 0x6D, 0x7E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SECTION_BG = RGBColor(0xE8, 0xF4, 0xF8)


def set_text(run, text, size=18, bold=False, color=DARK, font_name="Calibri"):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font_name


def add_title_slide(prs, data):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = ACCENT
    bg.line.fill.background()

    box = slide.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.5), Inches(1.2))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    set_text(r, data["title"], size=44, bold=True, color=WHITE)

    sub = slide.shapes.add_textbox(Inches(0.8), Inches(3.2), Inches(11.5), Inches(1.5))
    stf = sub.text_frame
    stf.word_wrap = True
    sp = stf.paragraphs[0]
    sp.alignment = PP_ALIGN.CENTER
    sr = sp.add_run()
    set_text(sr, data.get("subtitle", ""), size=20, color=WHITE)

    info = slide.shapes.add_textbox(Inches(0.8), Inches(5.0), Inches(11.5), Inches(2.0))
    itf = info.text_frame
    for i, line in enumerate(data["bullets"]):
        para = itf.paragraphs[0] if i == 0 else itf.add_paragraph()
        para.alignment = PP_ALIGN.CENTER
        para.space_after = Pt(6)
        ir = para.add_run()
        set_text(ir, line, size=16, color=WHITE)


def add_content_slide(prs, data):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    bar = slide.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(1.1))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.22), Inches(12.3), Inches(0.7))
    ttf = title_box.text_frame
    tp = ttf.paragraphs[0]
    tr = tp.add_run()
    set_text(tr, data["title"], size=28, bold=True, color=WHITE)

    y_start = Inches(1.35)
    if data.get("section"):
        sec = slide.shapes.add_shape(1, Inches(0.5), y_start, Inches(12.3), Inches(0.45))
        sec.fill.solid()
        sec.fill.fore_color.rgb = SECTION_BG
        sec.line.fill.background()
        sbox = slide.shapes.add_textbox(Inches(0.65), y_start + Inches(0.05), Inches(12), Inches(0.4))
        stf = sbox.text_frame
        sp = stf.paragraphs[0]
        sr = sp.add_run()
        set_text(sr, data["section"], size=14, bold=True, color=ACCENT)
        y_start += Inches(0.55)

    body = slide.shapes.add_textbox(Inches(0.65), y_start, Inches(12.0), Inches(5.8))
    btf = body.text_frame
    btf.word_wrap = True
    for i, bullet in enumerate(data["bullets"]):
        para = btf.paragraphs[0] if i == 0 else btf.add_paragraph()
        para.level = 0
        para.space_after = Pt(10)
        para.line_spacing = 1.15
        br = para.add_run()
        set_text(br, f"• {bullet}", size=17, color=DARK)


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    for slide_data in SLIDES:
        if slide_data.get("layout") == "title":
            add_title_slide(prs, slide_data)
        else:
            add_content_slide(prs, slide_data)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT))
    print(f"Saved {len(SLIDES)} slides to {OUTPUT}")


if __name__ == "__main__":
    build()

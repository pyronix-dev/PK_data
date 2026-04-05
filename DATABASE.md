# Drug Disposition PBPK Simulation Dataset

## Overview

This dataset contains synthetic but physiologically realistic pharmacokinetic (PK) data generated via a population PBPK simulator. It mimics Phase I clinical trial observations, tracking drug concentrations across various organs based on specific molecular and patient profiles.

---

## File Structure

The raw dataset consists of four primary files before any training/testing splits are applied:

- **data.csv** (~1,260,000 rows) — The master file. Contains all concentration curves and PK parameters merged together. This is the single source used to create public and private data splits.
- **molecules.csv** (~500 rows) — Molecular structures and computed descriptors. Mix of perturbed known drugs (~44%) and novel scaffolds (~56%).
- **patients.csv** (~200 rows) — Synthetic patient physiology and demographics, including realistic variability in organ function, genetics, and lab values.
- **all_curves.parquet** (~1,260,000 rows) — Raw concentration-time curves without the PK summary columns, stored in compact Parquet format for efficient loading and plotting.

---

## Data Schema Details

### 1. Molecular Structures (molecules.csv)

500 drug-like compounds with 16 physicochemical descriptors.

**Identifiers:** molecule_id, smiles (SMILES notation), drug_class.

**Physicochemical Properties:** Molecular weight (mw: 50–900 g/mol), logp (-2 to 7), tpsa (0–250 Å²), hydrogen bond donors (hbd), and acceptors (hba).

**Ionization & Binding:** pka_acidic, pka_basic, fraction_ionized_ph74, and fraction_unbound (plasma binding, 0.01–1.0).

**Drug Context:** morgan_fp_256 (256-bit hex), is_known_drug (FDA status), and similarity_to_known (Tanimoto similarity).

### 2. Patient Physiology (patients.csv)

200 synthetic profiles reflecting real-world physiological variability.

**Demographics:** age (18–85), sex (0=F, 1=M), weight_kg, height_cm, bmi, and bsa_m2.

**Organ Function:** hepatic_impairment (0.0 to 1.0 scale) and renal_impairment (creatinine clearance multiplier, 0.3–1.0).

**Genetics (CYP Enzymes):** Phenotype status for cyp2d6, cyp3a4, and cyp2c9:

- **PM** — Poor (2–10% prevalence, ~10% capacity)
- **IM** — Intermediate (10–15% prevalence, ~50% capacity)
- **NM** — Normal (70–85% prevalence, 100% capacity)
- **UM** — Ultra-rapid (1–10% prevalence, ~200% capacity)

**Labs & Flow:** albumin_gdl, alpha1_acid_glycoprotein, gastric_ph, hematocrit, and cardiac_output_l_min.

### 3. Combined Ground Truth (data.csv)

The primary training file. Each row represents a sample per compartment per time point.

**Context & Dosing:** sample_id, molecule_id, patient_id, dose_mg, and route (Oral or IV).

**Time Series:** compartment (Blood, Liver, Kidney, Brain, or Fat), time_h (0 to 72 hours), and concentration_ng_ml.

**PK Parameter Summaries** (repeated for every row of a sample):

- **Cmax/Tmax** — Peak concentration and time to peak.
- **AUC** — Area under the curve (0–72h and extrapolated to infinity).
- **Dynamics** — Half-life, total body clearance, and volume of distribution (vss_l).
- **Disposition** — Bioavailability, brain AUC ratio, fraction excreted in urine, and fraction metabolized.

---

## Methodology & Generation

The data is synthesized using a 7-compartment PBPK simulator.

- **Input:** 500 molecules and 200 patients are combined into 18,000 unique molecule-patient-dose combinations.
- **Mapping:** A QSPR (Quantitative Structure-Property Relationship) model maps molecular descriptors to mechanistic parameters like absorption rate (k_a), intrinsic clearance (CL_int), and tissue partitioning (K_p).
- **Simulation:** A system of Ordinary Differential Equations (ODE) covering the Gut, Liver, Blood, Kidney, Brain, Fat, and Muscle simulates drug flow.
- **Noise:** Realistic LC-MS/MS noise is added (10% analytical CV, 20% biological CV, 0.01 ng/mL floor).

### Key Dataset Properties

- **Mass Balance:** Total drug mass across all compartments does not exceed 1.5× the administered dose.
- **Physiological Plausibility:** All parameters fall within 3 standard deviations of published clinical population means.
- **Reproducibility:** All random seeds are fixed; re-running the generator with the same seed produces identical output.
- **LOD Floor:** All non-negative concentrations have a Limit of Detection floor of 0.01 ng/mL.

---

## Data Generation Scripts

The entire dataset is reproducible from scratch using the included Python scripts.

### `generate_data.py`

Main entry point for data generation. Combines molecules, patients, and the PBPK simulator to produce all output files.

**Usage:**

```bash
# Generate full dataset (all 18,000 samples)
python generate_data.py --mode all --seed 42

# Quick test run
python generate_data.py --mode all --seed 42 --quick

# Custom sizes
python generate_data.py --mode all --seed 42 --n-molecules 500 --n-patients 200 --n-samples 18000
```

**What it does:** Generates 500 molecules with RDKit-computed descriptors, 200 patients with realistic physiology, runs each combination through the PBPK ODE solver with QSPR-predicted parameters, adds LC-MS/MS noise, extracts PK parameters via NCA, and saves everything to `data.csv`, `molecules.csv`, `patients.csv`, and `all_curves.parquet`.

### `pbpk_simulator.py`

The physics engine. Contains four core components:

**PhysiologyModel** — Scales organ volumes and blood flows to patient size via allometric scaling. Applies hepatic/renal impairment by reducing blood flow and metabolic capacity. CYP genotypes (PM/IM/NM/UM) directly scale enzyme capacity.

**QSPRPredictor** — Maps molecular descriptors to mechanistic PK parameters using published structure-property relationships: absorption rate (k_a) from TPSA/logP/HBD count, intrinsic hepatic clearance (CL_int) from logP/MW/CYP status, tissue partitioning (K_p) from lipophilicity/pKa/ionization, fraction unbound (f_u) from logP/albumin levels, and bioavailability (F) from absorption fraction and first-pass metabolism.

**PBPKModel** — The 7-compartment ODE system (Gut, Liver, Blood, Kidney, Brain, Fat, Muscle). Each compartment is an organ connected by blood flow. Drug distributes into tissues based on partition coefficients, gets metabolized in the liver, and excreted by the kidneys. Solved with LSODA stiff solver at tight tolerances (rtol=1e-8, atol=1e-10). Mass is conserved by construction.

**PKParameterExtractor** — Performs Non-Compartmental Analysis on simulated curves: trapezoidal AUC, log-linear terminal half-life, clearance, volume of distribution, brain penetration ratio, and excretion fractions.

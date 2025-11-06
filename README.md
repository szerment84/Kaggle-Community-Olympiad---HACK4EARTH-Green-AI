# 🌱 Green AI Optimizer — Carbon-Aware Machine Learning
**Kaggle Community Olympiad — Hack4Earth Green AI Challenge**

---

## 🧠 Idea

This project demonstrates how to **reduce the carbon footprint of AI workloads**
by making machine-learning pipelines *carbon-aware* — automatically scheduling training
during hours of low-carbon electricity and measuring the CO₂ impact.

It provides two comparable pipelines:

- **Baseline Run** – trains a model with no carbon awareness (blind scheduling)  
- **Optimized Run** – automatically picks a *low-carbon-intensity window* before training  

Both runs are measured for runtime, energy proxy, and estimated CO₂ emissions.  
The difference between them demonstrates the potential for “green” AI scheduling.

---

## ⚙️ Repository Structure

├── data/
│ ├── train.csv
│ ├── test.csv
│ └── metaData.csv
│
├── report/
│ ├── GreenAI_Optimizer_Report.md
│ └── GreenAI_Optimizer_Report.pdf
│
├── src/
│ ├── pipeline.py # main training + measurement logic
│ ├── carbon_utils.py # carbon intensity & energy proxy utilities
│ └── impact.py # post-run impact analysis (annual CO₂ savings)
│
├── model-card.md
├── requirements.txt
├── run.sh # simple CLI launcher
├── README.md
└── LICENSE

---

## 🧩 Requirements

- Python ≥ 3.10  
- pandas, scikit-learn, numpy, tqdm  
- (optional) `codecarbon` for precise emission tracking  

Install dependencies:

```bash
pip install -r requirements.txt
🚀 Quick Start — Kaggle Notebook
Create a new Kaggle Notebook under Hack4Earth Green AI Challenge

Add dataset:
kaggle-community-olympiad-hack-4-earth-green-ai

Upload project files (src/, run.sh, requirements.txt)

Run:


pip install -r requirements.txt

# Baseline (no carbon awareness)
bash run.sh baseline /kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai

# Optimized (low-carbon window)
bash run.sh optimized /kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai
Equivalent Python commands:


python -m src.pipeline --mode baseline  --data /kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai
python -m src.pipeline --mode optimized --data /kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai
💻 Run Locally

git clone https://github.com/<your-username>/Kaggle-Community-Olympiad---HACK4EARTH-Green-AI.git
cd Kaggle-Community-Olympiad---HACK4EARTH-Green-AI

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.pipeline --mode baseline
python -m src.pipeline --mode optimized
python src/impact.py --metrics metrics_before_after.csv
📊 Output Files
File	Description
submission_baseline.csv	Model predictions in baseline mode
submission_optimized.csv	Predictions from optimized (low-CI) run
metrics_before_after.csv	Comparison of runtime, energy, CO₂, MAE
impact_report.csv	Annualized CO₂ savings (optional)

Example metrics file:

Scenario	Runtime_s	Energy_KWh	CO2e_kg	CO2_Reduction_%
Baseline	12.3	0.00034	0.00024	0.0
Optimized	10.1	0.00028	0.00018	25.0

🌍 Conceptual Flow
Detect Carbon Intensity Window → pick clean-energy hours

Execute Training Job → monitor runtime & energy proxy

Compare Scenarios → runtime vs. energy vs. CO₂

Integrate with OmniEnergy (optional) → industrial EMS / ISO 50001 compliance

🧾 Citation
If you use this repository or concept, please cite:

Szermet, M. (2025). Green AI Optimizer — Carbon-Aware Machine Learning.
Hack4Earth Green AI Challenge / Kaggle Community Olympiad.

🧰 License
MIT License © 2025 Martin Szermet

🪴 Links
🔗 Kaggle Notebook

💻 GitHub Repository

🏆 DoraHacks Hackathon Entry

⚡ OmniEnergy Integration

“Green AI is not about sacrificing intelligence — it’s about making intelligence responsible, measurable,
and aligned with the energy standards that will define the next decade of sustainable industry.”

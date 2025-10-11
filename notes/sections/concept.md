# Concept

- **Find a list of promising candidate cathode/anode materials for EV batteries 🔋**
- Relevant databases:
    - https://next-gen.materialsproject.org/
    - https://github.com/CambridgeMolecularEngineering/chemdataextractor2
    - https://www.materialsforbatteries.org/data/
    - https://batterymaterials.info/
- Potential approach: Use ML models to predict voltage, stability, and cycle life; validate via Materials Project Battery Explorer.
    - E.g: Use sentence-transformers to embed titles, DOIs, and properties from a curated battery dataset, store in FAISS for similarity search, retrieve top-k evidence at query time and pass to an LLM (via llama-cpp-python) to propose novel anode/cathode compositions with predicted capacity/voltage/citations in JSON format, augment with Materials Project data and raw material costs, deploy via Streamlit, or alternatively skip the LLM and train an XGBoost structure prediction model instead.

[This section will be enhanced by Oslo]

# Notebooks

Exploratory notebooks for the Frontiers submission go here, one per
experiment where a notebook is the natural format (e.g. plotting a
dose-response curve for E1, or SHAP dependence plots for E6).

The pipeline itself is not implemented in notebooks - `src/bigwig_qc/` and
`scripts/` are the source of truth, so that it's reproducible from the
command line / SLURM (see the top-level README). Notebooks here should
`import bigwig_qc` rather than redefining pipeline logic inline.

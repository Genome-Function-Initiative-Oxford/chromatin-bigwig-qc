# chromatin-bigwig-qc

**Machine learning–based quality control for chromatin accessibility bigWig tracks, grounded in biologically stable reference features.**

**chromatin-bigwig-qc** is a machine learning–based framework for quality control of chromatin accessibility bigWig signal tracks. The method operates directly on processed coverage files and integrates complementary signal-level metrics, including peak-level statistics, background noise estimation, and recovery of stable genomic reference features such as constant promoters and CTCF sites.

By leveraging an ML-based peak caller (LanceOtron) together with an interpretable XGBoost model, chromatin-bigwig-qc produces a continuous quality score that reflects a track’s ability to recover biologically meaningful regulatory and architectural elements of the genome. The framework is designed to support robust data integration, reliable downstream analyses, and machine learning applications in regulatory genomics.

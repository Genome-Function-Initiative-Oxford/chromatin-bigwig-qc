# chromatin-bigwig-qc
Machine learning–based quality control for chromatin accessibility bigWig tracks using biologically grounded reference features.

**chromatin-bigwig-qc** is a machine learning–based framework for quality control of chromatin accessibility bigWig tracks. The method operates directly on processed signal files and integrates peak-level statistics, background noise estimation, and recovery of stable genomic reference features such as constant promoters and CTCF sites.

Using an ML-based peak caller (LanceOtron) and an interpretable XGBoost model, bigwig-qc produces a continuous quality score that reflects a track’s ability to recover biologically meaningful regulatory and architectural elements. The framework is designed to support robust data integration, downstream analysis, and machine learning applications in regulatory genomics.

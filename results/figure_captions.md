# Figure captions (TV2, draft)

Numbers come from the files named after each caption; regenerate them with `python scripts/02_epochs_erp.py --step <step>`.
Values in [ ] still need checking against the final text.

**Figure 2. Ocular artefact removal with ICA (example participant).**
Five seconds of continuous EEG from sub-01 (0.1–40 Hz band-pass, recording reference FCz) at frontal channels Fp1, AFp2, AF7, AF8 and Fz, centred on a representative blink detected at Fp1 (median blink amplitude in this participant). Grey: before ICA; black: after removing the two ocular components shown in Figure 3 (IC0, blink/vertical eye movement; IC3, horizontal eye movement). Fp2 was a bad channel in this participant, so AFp2 is shown instead. Blink deflections of 100–200 µV at the frontopolar sites are removed while ongoing EEG is preserved.
*Sources: derivatives/v1/qc/sub-01_qc.json (ica_exclude), derivatives/v1/qc/sub-01-ica.fif. The pre-ICA data were rebuilt with the same functions as scripts/01_preprocess.py; after applying ICA, average reference and resampling, the rebuilt data matched the stored clean data to within 6e-5 µV.*

**Figure 3. Scalp topographies of the ocular independent components removed for sub-01.**
FastICA (30 components) fitted on a 1 Hz high-passed copy of the 0.1–40 Hz data. IC0: frontal, symmetric (blink/vertical eye movement); IC3: antisymmetric across the left and right frontal sites (horizontal eye movement). Full component properties (topography, ERP image, spectrum, variance) are shown in Supplementary Figures S_ICA_IC0 and S_ICA_IC3. Across participants, 2–4 ocular components were removed (mean 2.4 ± 0.7; Table S1).
*Sources: derivatives/v1/qc/sub-01_qc.json, results/table_S1_qc.csv.*

**Figure 4. Grand-average ERPs at Fz and Pz.**
Mean across participants of each participant's average ERP (equal weight per participant) for standard (grey, n = 42), target (red, n = 42) and distractor (blue dashed, n = 39) stimuli; shaded bands are bootstrap 95% confidence intervals across participants. Grey boxes mark the a priori measurement windows taken from the literature (P3a: Fz, 250–400 ms; P3b: Pz, 300–600 ms), chosen before looking at the difference waves. At Pz, targets elicited a large P3b (mean amplitude 7.79 ± 2.76 µV vs 1.33 ± 1.32 µV for standards; t(41) = 16.96, p < .001, dz = 2.62; median 50% fractional area latency 496 ms [IQR 481–520]). At Fz, no positive P3a was observed: in the P3a window, distractor ERPs were more negative than standard ERPs (−2.69 ± 2.39 vs −1.94 ± 1.68 µV; t(38) = −2.40, p = .021, dz = −0.39), consistent with a frontal N2 overlapping the window. Three participants who responded to most distractors (sub-02, 14, 17) are excluded from the distractor condition.
*Sources: results/erp_windows.json, results/erp_peaks_summary.csv, derivatives/v1/excluded.json.*

**Figure 5. Target − standard difference wave and cluster-based permutation test (n = 42).**
(a) Butterfly plot of the grand-average difference wave (all 127 channels, average reference) with scalp maps at fixed times 300, 400, 500 and 600 ms. (b) Channel × time image of the difference wave, channels grouped by region (frontal, central, parietal, occipital, temporal; grouping by channel-name prefix as in results/numbers_for_writing.md); samples outside significant clusters are masked. (c) Scalp maps at the same times and colour scale (±8.4 µV) as (a); channels belonging to a significant cluster at that time are marked. Spatio-temporal cluster-based permutation test (one-sample t-test on participants' difference waves, 0–800 ms, cluster-forming threshold two-tailed p < .05, 5000 permutations): one positive cluster (centro-parietal; 111 channels, 92–800 ms, p < .001) and one negative cluster (frontal and lateral; 107 channels, 124–800 ms, p < .001). Because of how cluster tests work, the extent of a cluster in time and space does not establish when or where the effect begins (Sassenhagen & Draschkow, 2019).
*Sources: results/erp_clusters.csv, results/cluster_target_minus_standard.npz, results/erp_peaks_summary.csv.*

**Supplementary Figure S (distractor − standard, n = 39).**
Same layout as Figure 5 for the distractor − standard difference wave: scalp maps at 300, 400, 500 and 600 ms, colour scale ±6.2 µV, channels grouped by region in the image. Clusters: one positive (117 channels, 96–800 ms, p < .001) and two negative (41 channels, 124–336 ms, p = .014; 95 channels, 300–800 ms, p < .001). The same caveat on cluster onset and extent applies.
*Sources: results/erp_clusters.csv, results/cluster_distractor_minus_standard.npz, results/erp_peaks_summary.csv.*

**Supplementary Figures S_ICA_IC0, S_ICA_IC3.** `ica.plot_properties` for the two ocular components removed in sub-01 (data: 1 Hz high-passed copy used for fitting).

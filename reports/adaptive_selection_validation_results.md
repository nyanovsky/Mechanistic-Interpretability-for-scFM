# Adaptive Gene Selection Validation Results

## Objective
Determine if adaptive gene selection (based on Participation Ratio) improves GO enrichment annotation success compared to the fixed 30-gene approach.

## Background
- **Current annotation:** 2115/3000 features (70.5%) with fixed 30 genes per feature
- **Hypothesis:** Fixed 30-gene selection fails for high-PR features because it under-samples distributed activation patterns
- **Approach:** Test adaptive selection on 150 non-significant features stratified by PR range

## Methodology

### Baseline (Fixed-30 Approach)
From existing annotations:
```
PR Range     Success    Total    Rate
<5           -          0        -
5-20         5          5        100.0%
20-100       1691       2364     71.5%
100-500      405        611      66.3%
>500         14         20       70.0%
-------------------------------------------
Overall      2115       3000     70.5%
```

**Key observation:** High-PR features (100-500) have the **lowest success rate** (66.3%) despite being common.

### Test Sample
- **150 non-significant features** sampled from 885 total non-significant
- **Stratified by PR:** 113 from PR 20-100, 35 from PR 100-500, 2 from PR >500
- **Method:** Re-run GO enrichment with adaptive gene selection

### Adaptive Parameters
- `scale_factor = 0.6` (select 60% of PR value)
- `min_genes = 10`
- `max_genes = 100`
- Gene filtering: mean_expr > 0.01 OR expressed in > 0.5% cells

**Adaptive gene counts by PR:**
- PR 20-100 → 40 genes (vs fixed 30) [+33%]
- PR 100-500 → 88 genes (vs fixed 30) [+193%]
- PR >500 → 100 genes (vs fixed 30) [+233%]

## Results

### Flip Rate (Non-significant → Significant)
```
Overall:      62/150 flipped (41.3%)

By PR range:
  PR 20-100:   36/113 flipped (31.9%)
  PR 100-500:  24/35 flipped (68.6%)
  PR >500:     2/2 flipped (100.0%)
```

### Key Findings

1. **Strong overall effect:** 41.3% of non-significant features became significant with adaptive selection

2. **PR-dependent improvement:**
   - Low-moderate PR (20-100): 31.9% flip rate - modest improvement
   - **High PR (100-500): 68.6% flip rate** - **very strong improvement**
   - Very high PR (>500): 100% flip rate - complete recovery

3. **The high-PR bottleneck is confirmed:** Features with PR 100-500 were severely limited by fixed 30-gene selection. They need 88-100 genes to capture sufficient biological signal.

## Projected Impact on Full Dataset

### Current State (Fixed-30)
- Total features analyzed: 3000
- Significant annotations: 2115 (70.5%)
- Non-significant annotations: 885 (29.5%)

### Projected with Adaptive Selection
- Expected additional annotations: 885 × 41.3% ≈ **365 features**
- New total: 2115 + 365 = **2480/3000 (82.7%)**
- **Improvement: +12.2 percentage points**

### By PR Range
Based on validation results:
```
PR Range    Current    Projected    Improvement
20-100      1691       1691 + 215   +215 features (+9.1%)
100-500     405        405 + 141    +141 features (+34.8%)
>500        14         14 + 4       +4 features (+28.6%)
```

The largest absolute gains come from PR 20-100 (most common), but the largest relative gains come from high-PR features (100-500).

## Interpretation

### Why Adaptive Works

**High-PR features have distributed activation:**
- Activation is spread across 100-500 genes
- Fixed 30 genes captures only a small fraction of the signal
- GO enrichment fails due to incomplete gene set

**Adaptive selection scales appropriately:**
- Allocates 88-100 genes for high-PR features
- Captures the full distributed pattern
- GO enrichment succeeds with complete biological context

**Example:** A feature representing "broad metabolic processes" might activate 200 genes moderately. Fixed-30 selection picks 30 random genes from this set, losing coherence. Adaptive-100 picks 100 genes, preserving the metabolic pathway structure.

### Why Some Features Still Fail

Even with adaptive selection, 58.7% of non-significant features remained non-significant. Possible reasons:

1. **Noise features:** No coherent biological function
2. **Novel/rare processes:** Not well-represented in GO databases
3. **Combinatorial features:** Represent interactions, not single pathways
4. **Threshold effects:** Close to significance but not quite (p ≈ 0.05-0.10)

## Recommendations

### 1. Adopt Adaptive Selection (Strongly Recommended)
Re-run full interpretation with adaptive selection enabled:
```bash
# Verify USE_ADAPTIVE_SELECTION = True in interpret_sae.py
python scripts/interpret_sae.py --layer 12 --expansion 8 --k 32 --n_features 3000
```

**Expected outcome:** ~82.7% annotation rate (2480/3000 features)

### 2. Parameter Tuning (Optional)
Current parameters are reasonable, but could be optimized:

**If high-PR features still struggle:**
- Increase `MAX_GENES_PER_FEATURE` from 100 to 150
- May help the PR >500 features

**If low-PR features are over-represented:**
- Decrease `MIN_GENES_PER_FEATURE` from 10 to 7
- Allows more focused features to use fewer genes

**Alternative scale factors:**
- Current: 0.6 (60% of PR)
- More conservative: 0.5 (50% of PR) - fewer genes, faster
- More aggressive: 0.8 (80% of PR) - more genes, slower

### 3. Apply to Other Layers
Once validated on layer 12, apply to layers 9 and 15:
```bash
python scripts/interpret_sae.py --layer 9 --expansion 8 --k 32 --n_features 3000
python scripts/interpret_sae.py --layer 15 --expansion 8 --k 32 --n_features 3000
```

### 4. Document Feature Scale in Summaries
When presenting results, include PR information:
- "Feature 1234 (PR=150): [GO terms]"
- Helps users understand feature granularity

## Computational Cost

**Fixed-30 approach:** 3000 features × 30 genes × 3 GO databases = ~270k API calls (estimated)

**Adaptive approach:** 3000 features × ~50 genes avg × 3 GO databases = ~450k API calls (estimated)

**Trade-off:** ~67% more API calls, but 12% more annotated features and much better biological validity for high-PR features.

## Conclusion

**Adaptive gene selection is necessary and effective.**

The fixed 30-gene approach systematically fails for high-PR features with distributed activation patterns. Adaptive selection recovers 41.3% of these features by appropriately scaling gene set size to feature granularity.

**Recommendation:** Adopt adaptive selection as the default method for all future SAE interpretations.

---

## Files Generated
- Test results: `/biodata/nyanovsky/datasets/pbmc3k/layer_12/sae_k_32_5120/adaptive_test_results/adaptive_test_results.csv`
- GO enrichment: `/biodata/nyanovsky/datasets/pbmc3k/layer_12/sae_k_32_5120/adaptive_test_results/feature_*_enrichr_adaptive/`

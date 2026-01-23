# Online SAE Training Design

## Problem Statement
Current workflow requires pre-extracting and storing all activations:
- PBMC3K (2,638 cells): ~130GB
- CELLxGENE 100k cells: ~5TB storage needed
- CELLxGENE 900k cells: ~45TB (infeasible)

**Solution**: Train SAE online by extracting activations and training in a single pass.

---

## Evolution of Strategy

### 1. Offline Training (The Baseline)
- **Workflow:** Generate Activations (to Disk) -> Train SAE (read from Disk).
- **Pros:** Statistically stable (global shuffling), maximizes GPU throughput during training.
- **Cons:** Requires massive storage (5TB+). I/O becomes the bottleneck.
- **Performance:** PBMC3K takes ~35 mins (excluding generation time).

### 2. Online "Naive" (HDF5 Buffer)
- **Workflow:** Generate Chunk -> Save HDF5 -> Train -> Delete.
- **Pros:** Low storage.
- **Cons:** Slow. Writes and Reads from disk add massive overhead.
- **Performance:** ~45 mins for PBMC3K.

### 3. Online "Minimal RAM" (Final Choice)
- **Workflow:** Generate Chunk (to RAM) -> Train (in RAM) -> Discard.
- **Pros:** Fastest. Zero Disk I/O. Safe Memory Usage.
- **Cons:** Requires "Local Epochs" approximation (training repeatedly on a chunk before moving on).
- **Mitigation:** Large chunk sizes (800+ cells) and global shuffling approximate global training well.
- **Performance:** ~37 mins for PBMC3K (includes generation time). **Faster than Offline when considering end-to-end time.**

---

## Comparison: RAM vs Disk (The "Hybrid" 1TB Idea)

A proposal was made to save 1TB chunks to disk to approximate offline training better.

**Why we rejected this:**
1. **Bottleneck is I/O:** Reading 1TB of float32 data 10 times (for 10 epochs) requires reading **10TB** of data. Even on fast NVMe (2GB/s), this adds ~1.5 hours of overhead per 1TB chunk.
2. **RAM is Faster:** Generating data in VRAM/RAM is often faster than reading it from disk.
3. **Double Penalty:** The hybrid approach pays the "Generation Cost" (to create the file) AND the "I/O Cost" (to read it back). Online training only pays the "Generation Cost".

**Conclusion:** Maximizing RAM usage (larger chunks) is superior to using Disk buffers.

---

## Final Architecture (Minimal RAM Strategy)

### Key Components

1.  **Global Shuffling:**
    -   We shuffle the list of all 100k cell indices *before* starting.
    -   This ensures Chunk 1 has the same statistical distribution as Chunk 500.

2.  **Memory Swapping (VRAM Optimization):**
    -   **Phase A (Extraction):** Move AIDO model to GPU. Extract batch. Move AIDO to CPU.
    -   **Phase B (Training):** Move SAE to GPU. Train on batch. Move SAE to CPU.
    -   **Benefit:** Keeps peak VRAM usage at ~12GB (max of one model) rather than ~22GB (sum of both).

3.  **Local Epochs:**
    -   Instead of 10 Global Epochs (reading dataset 10 times), we perform 10 Local Epochs (training on the current chunk 10 times).
    -   Valid because of Global Shuffling.

### Hyperparameters for 100k Cells
- **Chunk Size:** 800 cells (occupies ~38GB RAM). Fits in 70GB Cluster Node.
- **Validation:** Keep a static subset (e.g., 200 cells) in RAM for consistent metrics.

---

## Implementation Details

The script `scripts/train_sae_online_minimal.py` implements this optimized strategy.

### Usage
```bash
python scripts/train_sae_online_minimal.py --layer 12 --chunk_size 800
```

### Memory Math (100k Target)
- **1 Cell:** ~48 MB (19k genes x 640 dim x 4 bytes)
- **800 Cells:** ~38 GB
- **Validation (200 Cells):** ~9 GB
- **Total Peak RAM:** ~47 GB + Model Overhead (~5GB) = **~52 GB**.
- **Safety Margin:** ~18 GB (on 70GB node). Safe.

---
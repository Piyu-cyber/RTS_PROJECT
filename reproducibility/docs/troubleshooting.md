# Troubleshooting & FAQ

## 1. Environment & Dependencies

### Issue: `UnicodeEncodeError: 'charmap' codec can't encode character...`
- **Cause**: Windows PowerShell console default code page (cp1252) cannot print certain mathematical Unicode characters (e.g., $\cap, \emptyset$).
- **Solution**: The codebase uses clean ASCII descriptions in terminal logs (e.g. `intersect`, `empty`), and LaTeX KaTeX math in artifacts/docs. You can also run `$OutputEncoding = [Console]::InputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding` in PowerShell.

### Issue: `ModuleNotFoundError: No module named 'timm'`
- **Cause**: PyTorch Image Models library not installed.
- **Solution**: Install pinned dependencies:
  ```powershell
  pip install -r reproducibility/requirements.txt
  ```

---

## 2. Dataset & File Paths

### Issue: `FileNotFoundError: Manifest not found`
- **Cause**: Script 01 has not been executed yet.
- **Solution**: Run Step 01 to generate/verify manifests:
  ```powershell
  python reproducibility/scripts/01_prepare_dataset.py
  ```

### Issue: Raw BDD100K images not downloaded locally
- **Solution**: The simulation studies (Step 09) and metric analyses (Step 06, 07, 11) run deterministically from verified split manifests and saved prediction tensors without requiring all 100,000 raw image files. Hardware replay (Step 10) uses synthetic tensors if local image files are absent.

---

## 3. CUDA Graphs & GPU Execution

### Issue: `RuntimeError: CUDA error: operation not permitted` when capturing CUDA Graph
- **Cause**: Dynamic control flow, host-side synchronization, or tensor allocations inside the capture region.
- **Solution**: Static input and output buffers must be pre-allocated before `torch.cuda.graph()`. Ensure model is in `.eval()` mode and batch size is fixed to 1.

### Issue: CUDA Out of Memory (OOM) during full training
- **Cause**: Batch size 64 with full image resolution exceeds 8 GB VRAM.
- **Solution**: Use Automatic Mixed Precision (AMP `fp16`), which is enabled by default in `reproducibility/configs/training.yaml`, keeping peak VRAM below $5.4\text{ GB}$.

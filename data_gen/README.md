# OuroMamba-Gen
**OuroMamba-Gen** extends the internal dynamics of the **Mamba** architecture to enable *data-free calibration sample generation* for quantization.  
This module modifies Mamba’s state update mechanism to expose the **enhanced hidden state** `X_enhanced`, which aggregates local spatial context weighted by the per-token gate Δ(t), improving the implicit attention of Vision Mamba models.

---

## 🔍 Overview

Traditional Mamba implementations compress each token’s hidden state along the scanning direction, introducing a **local bias** that limits global token interaction.  
**OuroMamba-Gen** mitigates this by introducing *direction-agnostic spatial refinement*, effectively creating a more context-aware state representation.

### ✨ Key Features
- **Internal Mamba modification** — extends `selective_scan_ref` to compute and return the enhanced hidden state.  
- **Spatial State Fusion** — performs localized 3×3 neighborhood aggregation on token states, weighted by input gate Δ(t).  
- **Enhanced Implicit Attention** — yields refined hidden states (`X_enhanced`) that produce sharper, semantically aligned attention maps.

---

## 🧩 Integration

After modification, the internal scan function now exposes the enhanced hidden state:

```python
# Example usage
out, X_enhanced = selective_scan_ref(u, delta, A, B, C, return_last_state=False)
import time
import torch
import torch.nn.functional as F
import unittest
from torch.utils.cpp_extension import load
from src.quantizer import Q_Act
from custom_quantization_cuda import quantize_cuda, dequantize_cuda
                
def quantize_dequantize(static_s, s_outlier, inlier, outlier_coo, qmin, qmax, n_lva_o):
    """
    Given the outputs of extract_outliers, perform quantization and dequantization.
    This function now uses F.hardtanh for rounding and clamping, and for the outlier part
    it rounds in half-step increments to better match the original requant behavior.

    Parameters:
      static_s (torch.Tensor): Per-token static scaling factor for inlier quantization,
                               shape [B*L, 1].
      s_outlier (torch.Tensor): Per-token scaling factor for outlier quantization,
                                shape [B*L, 1].
      inlier (torch.Tensor): Inlier matrix with outlier positions zeroed, shape [B*L, D].
      outlier_coo (torch.Tensor): Sparse COO tensor for outliers with shape [B*L, D].
      qmin (int): Minimum quantization value (should match Q_Act.qmin).
      qmax (int): Maximum quantization value (should match Q_Act.qmax).
      n_lva_o (int): Number of quantization levels for outliers.

    Returns:
      x_reconstructed (torch.Tensor): The dequantized tensor after combining quantized inliers
                                      and outliers, shape [B*L, D].
    """
    # Compute outlier quantization limits.
    o_qmax = n_lva_o // 2 - 1
    o_qmin = -o_qmax

    # Quantize/dequantize inliers (using integer rounding).
    s_inlier = static_s.expand_as(inlier)
    inlier_q = torch.round(inlier / s_inlier)
    inlier_q = torch.clamp(inlier_q, min=qmin, max=qmax)
    inlier_dequant = inlier_q * s_inlier

    # Process outliers.
    indices = outlier_coo.indices()  # shape: [2, nnz]
    values = outlier_coo.values()    # shape: [nnz]
    
    if values.numel() > 0:
        # Pre-process outlier scales to have shape [nnz]
        s_outlier_vals = s_outlier[indices[0]].squeeze(1)
        # For outliers, round to the nearest 0.5 increment.
        outlier_q = torch.round(values / s_outlier_vals)
        outlier_q = torch.clamp(outlier_q, min=o_qmin, max=o_qmax)
        outlier_dequant_values = outlier_q * s_outlier_vals
        outlier_dequant = torch.sparse_coo_tensor(indices, outlier_dequant_values, outlier_coo.shape, device=inlier.device)
        outlier_dequant_dense = outlier_dequant.to_dense()
    else:
        outlier_dequant_dense = torch.zeros_like(inlier)
    
    x_reconstructed = inlier_dequant + outlier_dequant_dense
    return x_reconstructed




class TestQuantizer(unittest.TestCase):
    def setUp(self):
        # Set a manual seed for reproducibility and select device.
        torch.manual_seed(42)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.quantizer = Q_Act()
        # Set quantization bounds (not used in baseline).
        self.quantizer.qmin = -7
        self.quantizer.qmax = 7
        # Set number of outlier quantization levels.
        self.quantizer.n_lva_o = 8

    def debug_print(self, tag, tensor):
        print(f"{tag}: shape={tensor.shape}, mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}, min={tensor.min().item():.6f}, max={tensor.max().item():.6f}")

    def test_extract_outliers_vs_requant(self):
        test_cases = [{'B': 2, 'L': 20, 'D': 16, 'n_refresh': 10}]
        for case in test_cases:
            B, L, D, n_refresh = case['B'], case['L'], case['D'], case['n_refresh']
            with self.subTest(B=B, L=L, D=D, n_refresh=n_refresh):
                x = torch.randn(B, L, D, device=self.device) * 0.5
                # Inject outliers (~20% chance per token).
                for b in range(B):
                    for l in range(L):
                        if torch.rand(1).item() < 0.2:
                            num_outliers = max(1, D // 10)
                            outlier_indices = torch.randperm(D)[:num_outliers]
                            x[b, l, outlier_indices] *= 100.0
                static_s = torch.ones(L, device=self.device) * 0.001

                # Clone inputs.
                x_for_extract = x.clone()
                x_for_requant = x.clone()

                static_s_new, s_outlier, inlier, outlier_coo = self.quantizer.extract_outliers(
                    x_for_extract, n_refresh=n_refresh, static_o_list=None, static_s=static_s, qtype='uni'
                )
                print("[DEBUG] After extract_outliers (new method):")
                print(f"  static_s_new original shape: {static_s_new.shape}")
                print(f"  s_outlier shape: {s_outlier.shape}")
                print(f"  inlier shape: {inlier.shape}")
                outlier_coo = outlier_coo.to_sparse_coo()
                print(f"  outlier_coo nonzero count: {outlier_coo._nnz()}")

                static_s_new = static_s_new.expand(B, -1, -1).contiguous().view(B * L, 1)
                s_outlier = s_outlier.view(B * L, 1)
                inlier = inlier.half()

                x_quant_dequant = quantize_dequantize(
                    static_s_new, s_outlier, inlier, outlier_coo,
                    qmin=self.quantizer.qmin, qmax=self.quantizer.qmax, n_lva_o=self.quantizer.n_lva_o
                )
                print("[DEBUG] New method quantize-dequantize output:")
                self.debug_print("  x_quant_dequant", x_quant_dequant)

                x_requant_orig = self.quantizer.extracter_requant(
                    x_for_requant, n_refresh=n_refresh, static_o_list=None, static_s=static_s, qtype='uni'
                )
                x_requant_orig = x_requant_orig.view(B * L, D)
                print("[DEBUG] Original requant output:")
                self.debug_print("  x_requant_orig", x_requant_orig)

                diff = x_quant_dequant - x_requant_orig
                max_diff = diff.abs().max().item()
                mean_diff = diff.abs().mean().item()
                print(f"[DEBUG] Difference statistics: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

                self.assertTrue(torch.allclose(x_quant_dequant, x_requant_orig, rtol=1e-3, atol=1e-1),
                                f"Mismatch for B={B}, L={L}, D={D}, n_refresh={n_refresh}")

class TestCUDAQuantizer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.quantizer = Q_Act()
        self.quantizer.qmin = -7
        self.quantizer.qmax = 7
        self.quantizer.n_lva_o = 8

    def test_cuda_quant_dequant(self):
        B, L, D, n_refresh = 2, 20, 16, 10
        x = torch.randn(B, L, D, device=self.device) * 0.5
        # Inject outliers (~20% chance per token).
        for b in range(B):
            for l in range(L):
                if torch.rand(1).item() < 0.2:
                    num_outliers = max(1, D // 10)
                    outlier_indices = torch.randperm(D)[:num_outliers]
                    x[b, l, outlier_indices] *= 100.0
        static_s = torch.ones(L, device=self.device) * 0.001

        x_for_extract = x.clone()
        static_s_new, s_outlier, inlier, outlier_coo = self.quantizer.extract_outliers(
            x_for_extract, n_refresh=n_refresh, static_o_list=None, static_s=static_s, qtype='uni'
        )
        outlier_coo = outlier_coo.to_sparse_coo()
        static_s_new = static_s_new.expand(B, -1, -1).contiguous().view(B * L, 1)
        s_outlier = s_outlier.view(B * L, 1)
        # Cast inlier to FP16 for CUDA testing.
        inlier = inlier.half()
        s_inlier = static_s_new.view(B * L)
        # Pre-process outlier scales: extract the per-outlier scaling factors.
        outlier_indices = outlier_coo.indices()  # shape: [2, nnz]
        s_outlier_vals = s_outlier[outlier_indices[0]].squeeze(1)

        num_inlier_elements = inlier.numel()
        inlier_quantized_size = (num_inlier_elements + 1) // 2
        # Allocate quantized buffer as int8.
        quantized_inlier = torch.empty(inlier_quantized_size, device=self.device, dtype=torch.int8)
        nnz = outlier_coo._nnz()
        outlier_coo = outlier_coo.coalesce()
        # Force conversion: ensure outlier_values memory is truly FP16.
        outlier_values = outlier_coo.values().clone().half().contiguous().view(-1).to(self.device)
        print("outlier_values dtype:", outlier_values.dtype)

        quantized_outlier = torch.empty(nnz, device=self.device, dtype=torch.int8)
        x_reconstructed_cuda = torch.empty_like(inlier)

        # Call the CUDA quantization kernels.
        quantize_cuda(
            inlier, s_inlier,
            self.quantizer.qmin, self.quantizer.qmax,
            quantized_inlier, outlier_values, outlier_indices,
            s_outlier_vals, self.quantizer.n_lva_o, quantized_outlier
        )
        dequantize_cuda(
            quantized_inlier, s_inlier,
            B * L, D,
            quantized_outlier, outlier_indices,
            s_outlier_vals, nnz, x_reconstructed_cuda
        )
        torch.cuda.synchronize()

        x_quant_dequant = quantize_dequantize(
            static_s_new, s_outlier, inlier, outlier_coo,
            qmin=self.quantizer.qmin, qmax=self.quantizer.qmax, n_lva_o=self.quantizer.n_lva_o
        )
        diff = x_reconstructed_cuda - x_quant_dequant
        max_diff = diff.abs().max().item()
        self.assertTrue(torch.allclose(x_reconstructed_cuda.float(), x_quant_dequant, rtol=1e-3, atol=1e-1),
                        f"Mismatch in CUDA quant-dequant results: max_diff={max_diff}")

if __name__ == '__main__':
    unittest.main()

import os
import ctypes
from time import time
import numpy as np
import tvm
from tvm.script import tir as T
from tvm.ir import IRModule

from register import _posit_registered
# from tir_quire_matmul_extern import inject_quire_matmul_extern

_posit_registered()

LIBTVM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../build/libtvm.so"))


def _load_posit_converters():
    """Create numpy vectorized helpers for posit16(es=1) conversions."""
    libtvm = ctypes.CDLL(LIBTVM_PATH)

    libtvm.FloatToPosit16es1.restype = ctypes.c_uint16
    libtvm.FloatToPosit16es1.argtypes = [ctypes.c_float]
    float_to_posit16_es1 = np.vectorize(lambda x: libtvm.FloatToPosit16es1(np.float32(x)), otypes=[np.uint16])

    libtvm.Posit16es1ToFloat.restype = ctypes.c_float
    libtvm.Posit16es1ToFloat.argtypes = [ctypes.c_uint16]
    posit16_es1_to_float = np.vectorize(lambda x: libtvm.Posit16es1ToFloat(np.uint16(x)), otypes=[np.float32])

    return float_to_posit16_es1, posit16_es1_to_float

@T.prim_func
def matmul_float32(
    A: T.Buffer((T.int64(1), T.int64(768)), "float32"),
    B: T.Buffer((T.int64(768), T.int64(2304)), "float32"),
    C: T.Buffer((T.int64(1), T.int64(2304)), "float32"),
):
    """Float32 matmul with explicit loops showing multiple rounding"""
    for i1 in T.parallel(T.int64(2304)):
        for i0 in T.unroll(T.int64(1)):
            for k in range(T.int64(768)):
                with T.block("matmul"):
                    v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                    T.reads(A[v_i0, v_k], B[v_k, v_i1])
                    T.writes(C[v_i0, v_i1])
                    with T.init():
                        C[v_i0, v_i1] = T.float32(0)
                    C[v_i0, v_i1] = C[v_i0, v_i1] + A[v_i0, v_k] * B[v_k, v_i1]

@T.prim_func
def matmul(
    A: T.Buffer((T.int64(1), T.int64(768)), "custom[posites1]16"),
    B: T.Buffer((T.int64(768), T.int64(2304)), "custom[posites1]16"),
    C: T.Buffer((T.int64(1), T.int64(2304)), "custom[posites1]16"),
):
    for i1 in T.parallel(T.int64(2304)):
        for i0 in T.unroll(T.int64(1)):
            for k in range(T.int64(768)):
                with T.block("matmul"):
                    v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                    T.reads(A[v_i0, v_k], B[v_k, v_i1])
                    T.writes(C[v_i0, v_i1])
                    with T.init():
                        C[v_i0, v_i1] = T.cast(0, "custom[posites1]16")
                    C[v_i0, v_i1] = C[v_i0, v_i1] + A[v_i0, v_k] * B[v_k, v_i1]

@T.prim_func
def matmul_quire(
    A: T.Buffer((T.int64(1), T.int64(768)), "custom[posites1]16"),
    B: T.Buffer((T.int64(768), T.int64(2304)), "custom[posites1]16"),
    C: T.Buffer((T.int64(1), T.int64(2304)), "custom[posites1]16"),
):
    T.func_attr({"tir.noalias": True})
    T.evaluate(
        T.call_extern(
            "int32", 
            "Posit16es1QuireMatmul", 
            A.data, T.int64(1), T.int64(768), 
            B.data, T.int64(768), T.int64(2304), 
            C.data
        )
    )

@T.prim_func
def matmul2(
    A: T.Buffer((T.int64(1), T.int64(768)), "custom[posites1]16"),
    B: T.Buffer((T.int64(768), T.int64(2304)), "custom[posites1]16"),
    C: T.Buffer((T.int64(1), T.int64(2304)), "custom[posites1]16"),
):
    T.func_attr({"tir.noalias": True})
    
    for i1 in T.parallel(T.int64(2304)):
        for i0 in T.unroll(T.int64(1)):
            with T.block("matmul_quire_col"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(A[v_i0, 0:T.int64(768)], B[0:T.int64(768), v_i1])
                T.writes(C[v_i0, v_i1])
                T.call_extern(
                    "int32",
                    "Posit16es1QuireMatmulElem",
                    A.data,
                    T.int64(768),
                    B.data,
                    v_i1,
                    T.int64(2304),
                    C.data,
                    v_i0 * T.int64(2304) + v_i1
                )


def run_matmul_tests(seed: int = 0):
    """Test both standard matmul and QuireMatmul implementations"""
    print("="*80)
    print("POSIT16ES1 MATRIX MULTIPLICATION TEST")
    print("="*80)
    
    # Load posit converters
    float_to_posit, posit_to_float = _load_posit_converters()
    dev = tvm.cpu(0)

    # Test matrix dimensions
    M, K, N = 1, 768, 2304
    print(f"\nMatrix dimensions: A({M}x{K}) @ B({K}x{N}) = C({M}x{N})")
    
    # Generate random test matrices
    rng = np.random.default_rng(seed)
    A_f = (rng.standard_normal((M, K)).astype(np.float32) * 0.1).astype(np.float32)
    B_f = (rng.standard_normal((K, N)).astype(np.float32) * 0.1).astype(np.float32)
    
    print(f"Random seed: {seed}")
    print(f"A range: [{A_f.min():.4f}, {A_f.max():.4f}]")
    print(f"B range: [{B_f.min():.4f}, {B_f.max():.4f}]")

    # Convert float to posit16es1
    print("\nConverting float32 -> posit16es1...")
    A_u16 = float_to_posit(A_f).astype(np.uint16)
    B_u16 = float_to_posit(B_f).astype(np.uint16)
    
    # Initialize output buffers (as uint16, will be interpreted as posit16es1)
    C_u16_matmul = np.zeros((M, N), dtype=np.uint16)
    C_u16_quire = np.zeros((M, N), dtype=np.uint16)
    C_u16_matmul2 = np.zeros((M, N), dtype=np.uint16)

    # Create TVM arrays
    # First create uint16 arrays and copy data
    A_uint = tvm.runtime.empty((M, K), dtype="uint16", device=dev)
    B_uint = tvm.runtime.empty((K, N), dtype="uint16", device=dev)
    C_matmul_uint = tvm.runtime.empty((M, N), dtype="uint16", device=dev)
    C_quire_uint = tvm.runtime.empty((M, N), dtype="uint16", device=dev)
    C_matmul2_uint = tvm.runtime.empty((M, N), dtype="uint16", device=dev)
    
    A_uint.copyfrom(A_u16)
    B_uint.copyfrom(B_u16)
    C_matmul_uint.copyfrom(C_u16_matmul)
    C_quire_uint.copyfrom(C_u16_quire)
    C_matmul2_uint.copyfrom(C_u16_matmul2)
    
    # Create posit dtype arrays
    A_tvm = tvm.runtime.empty((M, K), dtype="custom[posites1]16", device=dev)
    B_tvm = tvm.runtime.empty((K, N), dtype="custom[posites1]16", device=dev)
    C_matmul_tvm = tvm.runtime.empty((M, N), dtype="custom[posites1]16", device=dev)
    C_quire_tvm = tvm.runtime.empty((M, N), dtype="custom[posites1]16", device=dev)
    C_matmul2_tvm = tvm.runtime.empty((M, N), dtype="custom[posites1]16", device=dev)
    
    # Copy uint16 data to posit arrays
    A_uint.copyto(A_tvm)
    B_uint.copyto(B_tvm)
    C_matmul_uint.copyto(C_matmul_tvm)
    C_quire_uint.copyto(C_quire_tvm)
    C_matmul2_uint.copyto(C_matmul2_tvm)
    
    # Create Float32 TVM arrays for comparison
    A_f32_tvm = tvm.runtime.empty((M, K), dtype="float32", device=dev)
    B_f32_tvm = tvm.runtime.empty((K, N), dtype="float32", device=dev)
    C_f32_tvm = tvm.runtime.empty((M, N), dtype="float32", device=dev)
    A_f32_tvm.copyfrom(A_f.astype(np.float32))
    B_f32_tvm.copyfrom(B_f.astype(np.float32))

    # Build all functions (including float32 matmul)
    mod = IRModule({
        "matmul_float32": matmul_float32,
        "matmul": matmul, 
        "matmul_quire": matmul_quire, 
        "matmul2": matmul2
    })
    print("\nBuilding TVM module (target=llvm)...")
    lib = tvm.build(mod, target="llvm")
    print("Build finished successfully!")

    # Get functions
    f_matmul_float32 = lib.get_function("matmul_float32")
    f_matmul = lib.get_function("matmul")
    f_quire = lib.get_function("matmul_quire")
    f_matmul2 = lib.get_function("matmul2")

    # Run Float32 matmul (to show Float32 also has rounding error)
    print("\n" + "-"*80)
    print("Running Float32 matmul (768 * 2 = 1536 rounding operations per element)...")
    start_time = time()
    f_matmul_float32(A_f32_tvm, B_f32_tvm, C_f32_tvm)
    end_time = time()
    print(f"Float32 matmul completed in {end_time - start_time:.6f} seconds!")
    print("-"*80)
    
    # Run standard matmul
    print("\n" + "-"*80)
    print("Running standard matmul (posit multiply-add)...")
    start_time = time()
    f_matmul(A_tvm, B_tvm, C_matmul_tvm)
    end_time = time()
    print(f"Standard matmul completed in {end_time - start_time:.6f} seconds!")
    print("-"*80)
    
    # Run QuireMatmul
    print("\n" + "-"*80)
    print("Running QuireMatmul (TRUE quire accumulation)...")
    start_time = time()
    f_quire(A_tvm, B_tvm, C_quire_tvm)
    end_time = time()
    print(f"QuireMatmul completed in {end_time - start_time:.6f} seconds!")
    print("-"*80)
    
    # Run matmul2 (loop structure with QuireMatmulRow)
    print("\n" + "-"*80)
    print("Running matmul2 (loop + QuireMatmulRow extern)...")
    start_time = time()
    f_matmul2(A_tvm, B_tvm, C_matmul2_tvm)
    end_time = time()
    print(f"matmul2 completed in {end_time - start_time:.6f} seconds!")
    print("-"*80)

    # Convert results back to float for comparison
    print("\nConverting posit16es1 -> float32...")
    # Copy posit results to uint16 arrays for conversion
    C_matmul_out_uint = tvm.runtime.empty((M, N), dtype="uint16", device=dev)
    C_quire_out_uint = tvm.runtime.empty((M, N), dtype="uint16", device=dev)
    C_matmul2_out_uint = tvm.runtime.empty((M, N), dtype="uint16", device=dev)
    
    C_matmul_tvm.copyto(C_matmul_out_uint)
    C_quire_tvm.copyto(C_quire_out_uint)
    C_matmul2_tvm.copyto(C_matmul2_out_uint)
    
    C_matmul_f = posit_to_float(C_matmul_out_uint.numpy()).astype(np.float32)
    C_quire_f = posit_to_float(C_quire_out_uint.numpy()).astype(np.float32)
    C_matmul2_f = posit_to_float(C_matmul2_out_uint.numpy()).astype(np.float32)
    
    # Get Float32 TVM result
    C_f32_tvm_result = C_f32_tvm.numpy()
    
    # Compute HIGH PRECISION reference using Float64
    print("Computing HIGH PRECISION reference (float64 - closer to mathematical truth)...")
    C_ref_f64 = np.matmul(A_f.astype(np.float64), B_f.astype(np.float64))
    print(f"Float64 reference range: [{C_ref_f64.min():.10f}, {C_ref_f64.max():.10f}]")
    
    # Also compute Float32 numpy reference (for comparison with TVM Float32)
    C_ref_f32 = np.matmul(A_f.astype(np.float32), B_f.astype(np.float32))
    print(f"Float32 numpy range: [{C_ref_f32.min():.4f}, {C_ref_f32.max():.4f}]")

    # Compare results
    print("\n" + "="*80)
    print("ACCURACY ANALYSIS (vs Float64 High Precision Reference)")
    print("="*80)
    print("\nThis shows that ALL methods have rounding errors!")
    print("Float64 is our 'ground truth' (closest to mathematical true value)\n")
    
    def compute_metrics(name, out_f, reference=C_ref_f64):
        """Compute and print accuracy metrics against reference"""
        diff = out_f.astype(np.float64) - reference
        abs_err = np.abs(diff)
        rel_err = abs_err / (np.abs(reference) + 1e-10)
        
        print(f"\n{name}:")
        print(f"  Output range: [{out_f.min():.10f}, {out_f.max():.10f}]")
        print(f"  Max absolute error vs Float64: {abs_err.max():.10e}")
        print(f"  Mean absolute error vs Float64: {abs_err.mean():.10e}")
        print(f"  Max relative error: {rel_err.max():.6e}")
        print(f"  Mean relative error: {rel_err.mean():.6e}")
        
        return abs_err, rel_err

    # Float32 TVM matmul accuracy (show Float32 also has error!)
    print("\n" + "-"*80)
    print("🔍 FLOAT32 ROUNDING ERROR DEMONSTRATION")
    print("-"*80)
    abs_err_f32_tvm, rel_err_f32_tvm = compute_metrics("Float32 TVM matmul (1536 roundings/element)", C_f32_tvm_result)
    
    # Float32 numpy reference (for comparison)
    abs_err_f32_numpy, rel_err_f32_numpy = compute_metrics("Float32 Numpy matmul (also has rounding)", C_ref_f32)
    
    print("\n⚠️  KEY INSIGHT: Float32 has cumulative rounding error!")
    print(f"   Even Float32 differs from Float64 'truth' by up to {abs_err_f32_tvm.max():.6e}")
    print("   This is because each multiply + add rounds to 23-bit mantissa\n")
    
    # Standard matmul accuracy
    print("\n" + "-"*80)
    print("POSIT16 IMPLEMENTATIONS (vs Float64 Reference)")
    print("-"*80)
    abs_err_matmul, rel_err_matmul = compute_metrics("Posit16 Standard Matmul (no Quire)", C_matmul_f)
    
    # QuireMatmul accuracy
    abs_err_quire, rel_err_quire = compute_metrics("QuireMatmul (TRUE quire)", C_quire_f)
    
    # matmul2 accuracy
    abs_err_matmul2, rel_err_matmul2 = compute_metrics("matmul2 (loop + QuireMatmulRow)", C_matmul2_f)
    
    # Comparison
    print("\n" + "="*80)
    print("PRECISION COMPARISON (All vs Float64 'Mathematical Truth')")
    print("="*80)
    
    # 0. Float32 rounding error
    print("\n[0] Float32 Cumulative Rounding Error:")
    print(f"  Float32 TVM max error: {abs_err_f32_tvm.max():.10e}")
    print(f"  Float32 Numpy max error: {abs_err_f32_numpy.max():.10e}")
    print(f"  ⚠️  Float32 is NOT perfect! It has {abs_err_f32_tvm.max():.3e} cumulative error")
    print(f"  This is from 768 * 2 = 1536 rounding operations per output element")
    
    # 1. Posit16 implementations comparison (with/without quire)
    print("\n[1] Posit16 Quire vs Posit16 Standard (both vs Float64 truth):")
    improvement_max = abs_err_matmul.max() / (abs_err_quire.max() + 1e-15)
    improvement_mean = abs_err_matmul.mean() / (abs_err_quire.mean() + 1e-15)
    print(f"  QuireMatmul max error improvement: {improvement_max:.2f}x")
    print(f"  QuireMatmul mean error improvement: {improvement_mean:.2f}x")
    
    improvement_max2 = abs_err_matmul.max() / (abs_err_matmul2.max() + 1e-15)
    improvement_mean2 = abs_err_matmul.mean() / (abs_err_matmul2.mean() + 1e-15)
    print(f"  matmul2 max error improvement: {improvement_max2:.2f}x")
    print(f"  matmul2 mean error improvement: {improvement_mean2:.2f}x")
    
    # 2. ALL methods vs Float64 truth
    print("\n[2] Error Distance from Mathematical Truth (Float64):")
    print(f"  Float32 TVM:          {abs_err_f32_tvm.max():.10e}")
    print(f"  Float32 Numpy:        {abs_err_f32_numpy.max():.10e}")
    print(f"  Posit16 no Quire:     {abs_err_matmul.max():.10e}")
    print(f"  Posit16 with Quire:   {abs_err_quire.max():.10e}")
    print(f"  Posit16 matmul2:      {abs_err_matmul2.max():.10e}")
    
    # 3. Key insight: Quire reduces gap to Float32
    quire_vs_f32_ratio = abs_err_quire.max() / abs_err_f32_tvm.max()
    no_quire_vs_f32_ratio = abs_err_matmul.max() / abs_err_f32_tvm.max()
    print("\n[3] How close to Float32 precision?")
    print(f"  Posit16 no Quire:   {no_quire_vs_f32_ratio:.1f}x worse than Float32")
    print(f"  Posit16 with Quire: {quire_vs_f32_ratio:.1f}x worse than Float32")
    print(f"  Quire improvement:  Makes Posit16 {no_quire_vs_f32_ratio/quire_vs_f32_ratio:.1f}x closer to Float32!")
    
    # 3. Quire benefit: how much closer to Float32 precision
    quire_benefit_max = (abs_err_matmul.max() - abs_err_quire.max()) / abs_err_matmul.max() * 100
    quire_benefit_mean = (abs_err_matmul.mean() - abs_err_quire.mean()) / abs_err_matmul.mean() * 100
    print(f"\n[4] Quire Benefit (error reduction):")
    print(f"  Max error reduced by: {quire_benefit_max:.1f}%")
    print(f"  Mean error reduced by: {quire_benefit_mean:.1f}%")
    
    # 4. Check if QuireMatmul and matmul2 produce identical results
    quire_diff = np.abs(C_quire_f - C_matmul2_f)
    print(f"\n[5] QuireMatmul vs matmul2 difference:")
    print(f"  Max difference: {quire_diff.max():.6e}")
    print(f"  Mean difference: {quire_diff.mean():.6e}")
    
    # Test passed/failed
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    
    # Check if results are reasonable (within expected posit16 precision)
    matmul_passed = abs_err_matmul.max() < 1.0  # Adjust threshold as needed
    quire_passed = abs_err_quire.max() < 1.0
    matmul2_passed = abs_err_matmul2.max() < 1.0
    
    print(f"\nStandard Matmul: {'✓ PASSED' if matmul_passed else '✗ FAILED'}")
    print(f"QuireMatmul: {'✓ PASSED' if quire_passed else '✗ FAILED'}")
    print(f"matmul2: {'✓ PASSED' if matmul2_passed else '✗ FAILED'}")
    
    if matmul_passed and quire_passed and matmul2_passed:
        print("\n🎉 Test PASSED! All implementations are correct.")
        if improvement_max > 1.5:
            print(f"✨ QuireMatmul provides {improvement_max:.1f}x better precision!")
        if quire_diff.max() < 1e-10:
            print(f"✨ matmul2 produces identical results to QuireMatmul!")
    else:
        print("\n⚠️  Test FAILED. Please check the implementation.")
    
    print("="*80)


# ============================================================================
# 3D Matmul Tests (for testing TIR pass transformation)
# ============================================================================

@T.prim_func
def matmul_3d_original(
    A: T.Buffer((T.int64(1), T.int64(1), T.int64(768)), "custom[posites1]16"),
    B: T.Buffer((T.int64(768), T.int64(50257)), "custom[posites1]16"),
    C: T.Buffer((T.int64(1), T.int64(1), T.int64(50257)), "custom[posites1]16"),
):
    """3D matmul: (1, 1, 768) @ (768, 50257) -> (1, 1, 50257)"""
    T.func_attr({"tir.noalias": True})
    for i2 in T.parallel(T.int64(50257)):
        for i0 in T.unroll(T.int64(1)):
            for i1, k in T.grid(T.int64(1), T.int64(768)):
                with T.block("matmul"):
                    v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                    T.reads(A[v_i0, v_i1, v_k], B[v_k, v_i2])
                    T.writes(C[v_i0, v_i1, v_i2])
                    with T.init():
                        C[v_i0, v_i1, v_i2] = T.Cast("custom[posites1]16", 0.0)
                    C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_k, v_i2]


@T.prim_func
def matmul_4d_original(
    A: T.Buffer((T.int64(1), T.int64(12), T.int64(1), T.int64(64)), "custom[posites1]16"),
    B: T.Buffer((T.int64(1), T.int64(12), T.int64(64), T.int64(64)), "custom[posites1]16"),
    C: T.Buffer((T.int64(1), T.int64(12), T.int64(1), T.int64(64)), "custom[posites1]16"),
):
    """4D matmul: (1, 12, 1, 64) @ (1, 12, 64, 64) -> (1, 12, 1, 64)"""
    T.func_attr({"tir.noalias": True})
    for i3 in T.parallel(T.int64(64)):
        for i1 in T.unroll(T.int64(12)):
            for i0, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(64)):
                with T.block("matmul"):
                    v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                    T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                    T.writes(C[v_i0, v_i1, v_i2, v_i3])
                    with T.init():
                        C[v_i0, v_i1, v_i2, v_i3] = T.Cast("custom[posites1]16", 0.0)
                    C[v_i0, v_i1, v_i2, v_i3] = C[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]


def test_3d_4d_matmul_pass(seed: int = 0):
    """Test TIR pass transformation on 3D and 4D matmul"""
    print("="*80)
    print("3D & 4D MATMUL TIR PASS TRANSFORMATION TEST")
    print("="*80)
    
    from tir_transform_matmul_to_quire import InjectQuireMatmulElem
    
    # Load posit converters
    float_to_posit, posit_to_float = _load_posit_converters()
    dev = tvm.cpu(0)
    
    # ========================================================================
    # Test 1: 3D Matmul (1, 1, 768) @ (768, 50257) -> (1, 1, 50257)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: 3D Matmul Transformation")
    print("="*80)
    
    # Use smaller dimensions for testing
    B, M, K, N = 1, 1, 768, 2304  # Reduced N for faster testing
    print(f"\nDimensions: A({B}, {M}, {K}) @ B({K}, {N}) = C({B}, {M}, {N})")
    
    # Generate test data
    rng = np.random.default_rng(seed)
    A_3d_f = (rng.standard_normal((B, M, K)).astype(np.float32) * 0.1)
    B_2d_f = (rng.standard_normal((K, N)).astype(np.float32) * 0.1)
    
    # Convert to posit
    A_3d_u16 = float_to_posit(A_3d_f).astype(np.uint16)
    B_2d_u16 = float_to_posit(B_2d_f).astype(np.uint16)
    
    # Create TVM arrays
    A_3d_tvm = tvm.runtime.empty((B, M, K), dtype="custom[posites1]16", device=dev)
    B_2d_tvm = tvm.runtime.empty((K, N), dtype="custom[posites1]16", device=dev)
    C_3d_orig_tvm = tvm.runtime.empty((B, M, N), dtype="custom[posites1]16", device=dev)
    C_3d_trans_tvm = tvm.runtime.empty((B, M, N), dtype="custom[posites1]16", device=dev)
    
    # Copy data via uint16
    A_3d_uint = tvm.runtime.empty((B, M, K), dtype="uint16", device=dev)
    B_2d_uint = tvm.runtime.empty((K, N), dtype="uint16", device=dev)
    A_3d_uint.copyfrom(A_3d_u16)
    B_2d_uint.copyfrom(B_2d_u16)
    A_3d_uint.copyto(A_3d_tvm)
    B_2d_uint.copyto(B_2d_tvm)
    
    # Build original 3D matmul
    mod_3d = IRModule({"matmul_3d": matmul_3d_original})
    
    # Create a version with adjusted dimensions
    @T.prim_func
    def matmul_3d_test(
        A: T.Buffer((T.int64(1), T.int64(1), T.int64(768)), "custom[posites1]16"),
        B: T.Buffer((T.int64(768), T.int64(2304)), "custom[posites1]16"),
        C: T.Buffer((T.int64(1), T.int64(1), T.int64(2304)), "custom[posites1]16"),
    ):
        T.func_attr({"tir.noalias": True})
        for i2 in T.parallel(T.int64(2304)):
            for i0 in T.unroll(T.int64(1)):
                for i1, k in T.grid(T.int64(1), T.int64(768)):
                    with T.block("matmul"):
                        v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                        T.reads(A[v_i0, v_i1, v_k], B[v_k, v_i2])
                        T.writes(C[v_i0, v_i1, v_i2])
                        with T.init():
                            C[v_i0, v_i1, v_i2] = T.Cast("custom[posites1]16", 0.0)
                        C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_k, v_i2]
    
    mod_3d_test = IRModule({"matmul_3d": matmul_3d_test})
    
    print("\nOriginal 3D matmul TIR:")
    print(mod_3d_test["matmul_3d"].script())
    
    # Apply transformation
    print("\nApplying InjectQuireMatmulElem pass...")
    with tvm.transform.PassContext(opt_level=0):
        mod_3d_transformed = InjectQuireMatmulElem()(mod_3d_test)
    
    print("\nTransformed 3D matmul TIR:")
    print(mod_3d_transformed["matmul_3d"].script())
    
    # Build both versions
    print("\nBuilding original and transformed modules...")
    lib_3d_orig = tvm.build(mod_3d_test, target="llvm")
    lib_3d_trans = tvm.build(mod_3d_transformed, target="llvm")
    
    # Run both
    print("\nRunning original 3D matmul...")
    f_3d_orig = lib_3d_orig.get_function("matmul_3d")
    f_3d_orig(A_3d_tvm, B_2d_tvm, C_3d_orig_tvm)
    
    print("Running transformed 3D matmul...")
    f_3d_trans = lib_3d_trans.get_function("matmul_3d")
    f_3d_trans(A_3d_tvm, B_2d_tvm, C_3d_trans_tvm)
    
    # Compare results
    C_3d_orig_uint = tvm.runtime.empty((B, M, N), dtype="uint16", device=dev)
    C_3d_trans_uint = tvm.runtime.empty((B, M, N), dtype="uint16", device=dev)
    C_3d_orig_tvm.copyto(C_3d_orig_uint)
    C_3d_trans_tvm.copyto(C_3d_trans_uint)
    
    C_3d_orig_f = posit_to_float(C_3d_orig_uint.numpy())
    C_3d_trans_f = posit_to_float(C_3d_trans_uint.numpy())
    
    diff_3d = np.abs(C_3d_orig_f - C_3d_trans_f)
    print(f"\n3D Matmul Result Comparison:")
    print(f"  Max difference: {diff_3d.max():.10e}")
    print(f"  Mean difference: {diff_3d.mean():.10e}")
    
    if diff_3d.max() < 1e-6:
        print("  ✅ 3D TRANSFORMATION PASSED!")
    else:
        print("  ❌ 3D TRANSFORMATION FAILED!")
        print(f"  Original output sample: {C_3d_orig_f.flatten()[:5]}")
        print(f"  Transformed output sample: {C_3d_trans_f.flatten()[:5]}")
    
    # ========================================================================
    # Test 2: 4D Matmul (1, 12, 1, 64) @ (1, 12, 64, 64) -> (1, 12, 1, 64)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: 4D Matmul Transformation")
    print("="*80)
    
    Batch, Heads, M4, K4, N4 = 1, 12, 1, 64, 64
    print(f"\nDimensions: A({Batch}, {Heads}, {M4}, {K4}) @ B({Batch}, {Heads}, {K4}, {N4}) = C({Batch}, {Heads}, {M4}, {N4})")
    
    # Generate test data
    A_4d_f = (rng.standard_normal((Batch, Heads, M4, K4)).astype(np.float32) * 0.1)
    B_4d_f = (rng.standard_normal((Batch, Heads, K4, N4)).astype(np.float32) * 0.1)
    
    # Convert to posit
    A_4d_u16 = float_to_posit(A_4d_f).astype(np.uint16)
    B_4d_u16 = float_to_posit(B_4d_f).astype(np.uint16)
    
    # Create TVM arrays
    A_4d_tvm = tvm.runtime.empty((Batch, Heads, M4, K4), dtype="custom[posites1]16", device=dev)
    B_4d_tvm = tvm.runtime.empty((Batch, Heads, K4, N4), dtype="custom[posites1]16", device=dev)
    C_4d_orig_tvm = tvm.runtime.empty((Batch, Heads, M4, N4), dtype="custom[posites1]16", device=dev)
    C_4d_trans_tvm = tvm.runtime.empty((Batch, Heads, M4, N4), dtype="custom[posites1]16", device=dev)
    
    # Copy data via uint16
    A_4d_uint = tvm.runtime.empty((Batch, Heads, M4, K4), dtype="uint16", device=dev)
    B_4d_uint = tvm.runtime.empty((Batch, Heads, K4, N4), dtype="uint16", device=dev)
    A_4d_uint.copyfrom(A_4d_u16)
    B_4d_uint.copyfrom(B_4d_u16)
    A_4d_uint.copyto(A_4d_tvm)
    B_4d_uint.copyto(B_4d_tvm)
    
    # Build original 4D matmul
    mod_4d = IRModule({"matmul_4d": matmul_4d_original})
    
    print("\nOriginal 4D matmul TIR:")
    print(mod_4d["matmul_4d"].script())
    
    # Apply transformation
    print("\nApplying InjectQuireMatmulElem pass...")
    with tvm.transform.PassContext(opt_level=0):
        mod_4d_transformed = InjectQuireMatmulElem()(mod_4d)
    
    print("\nTransformed 4D matmul TIR:")
    print(mod_4d_transformed["matmul_4d"].script())
    
    # Build both versions
    print("\nBuilding original and transformed modules...")
    lib_4d_orig = tvm.build(mod_4d, target="llvm")
    lib_4d_trans = tvm.build(mod_4d_transformed, target="llvm")
    
    # Run both
    print("\nRunning original 4D matmul...")
    f_4d_orig = lib_4d_orig.get_function("matmul_4d")
    f_4d_orig(A_4d_tvm, B_4d_tvm, C_4d_orig_tvm)
    
    print("Running transformed 4D matmul...")
    f_4d_trans = lib_4d_trans.get_function("matmul_4d")
    f_4d_trans(A_4d_tvm, B_4d_tvm, C_4d_trans_tvm)
    
    # Compare results
    C_4d_orig_uint = tvm.runtime.empty((Batch, Heads, M4, N4), dtype="uint16", device=dev)
    C_4d_trans_uint = tvm.runtime.empty((Batch, Heads, M4, N4), dtype="uint16", device=dev)
    C_4d_orig_tvm.copyto(C_4d_orig_uint)
    C_4d_trans_tvm.copyto(C_4d_trans_uint)
    
    C_4d_orig_f = posit_to_float(C_4d_orig_uint.numpy())
    C_4d_trans_f = posit_to_float(C_4d_trans_uint.numpy())
    
    diff_4d = np.abs(C_4d_orig_f - C_4d_trans_f)
    print(f"\n4D Matmul Result Comparison:")
    print(f"  Max difference: {diff_4d.max():.10e}")
    print(f"  Mean difference: {diff_4d.mean():.10e}")
    
    if diff_4d.max() < 1e-6:
        print("  ✅ 4D TRANSFORMATION PASSED!")
    else:
        print("  ❌ 4D TRANSFORMATION FAILED!")
        print(f"  Original output sample: {C_4d_orig_f.flatten()[:5]}")
        print(f"  Transformed output sample: {C_4d_trans_f.flatten()[:5]}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if diff_3d.max() < 1e-6 and diff_4d.max() < 1e-6:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80)
                                    

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    seed = 0
    test_mode = "2d"  # Default: test 2D matmul
    
    if len(sys.argv) > 1:
        test_mode = sys.argv[1].lower()
        if test_mode not in ["2d", "3d4d", "all"]:
            print(f"Invalid test mode: {test_mode}")
            print("Usage: python test.py [2d|3d4d|all] [seed]")
            print("  2d:   Test 2D matmul (default)")
            print("  3d4d: Test 3D and 4D matmul transformations")
            print("  all:  Run all tests")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            seed = int(sys.argv[2])
        except ValueError:
            print(f"Invalid seed value: {sys.argv[2]}, using default seed=0")
    
    print("POSIT16ES1 MATRIX MULTIPLICATION TEST")
    print("="*80)
    
    if test_mode == "2d" or test_mode == "all":
        print("Testing 2D matmul (standard matmul and QuireMatmul extern)")
        print()
        run_matmul_tests(seed=seed)
    
    if test_mode == "3d4d" or test_mode == "all":
        if test_mode == "all":
            print("\n\n")
        print("Testing 3D and 4D matmul TIR pass transformations")
        print()
        test_3d_4d_matmul_pass(seed=seed)

"""
TIR Pass: Transform standard matmul to QuireMatmulElem-based implementation

This pass converts standard matmul patterns (2D, 3D, 4D) to QuireMatmul extern calls.

2D matmul pattern (M=1, K, N):
    for i1 in parallel(N):
        for i0 in unroll(M=1):
            for k in range(K):  # reduction axis
                C[i0, i1] = C[i0, i1] + A[i0, k] * B[k, i1]
    
    Transforms to:
        for i1 in parallel(N):
            for i0 in unroll(M=1):
                call_extern("Posit{bits}es{es}QuireMatmul", A, offsetA, K, B, offsetB, i1, N, C, offsetC)

3D matmul pattern (batch, M=1, K, N):
    for i2 in parallel(N):
        for i0 in unroll(batch=1):
            for i1, k in grid(M=1, K):  # i1 spatial, k reduction
                C[i0, i1, i2] = C[i0, i1, i2] + A[i0, i1, k] * B[k, i2]
    
    Transforms to:
        for i2 in parallel(N):
            for i0 in unroll(batch=1):
                for i1 in grid(M=1):
                    call_extern("Posit{bits}es{es}QuireMatmul", A_flat, offsetA, K, B, offsetB, i2, N, C_flat, offsetC)

4D matmul pattern (B, H, M=1, K, N):
    for i3 in parallel(N):
        for i1 in unroll(H):
            for i0, i2, k in grid(B=1, M=1, K):  # k is reduction
                C[i0, i1, i2, i3] = C[i0, i1, i2, i3] + A[i0, i1, i2, k] * B[i0, i1, k, i3]
    
    Transforms to:
        for i3 in parallel(N):
            for i1 in unroll(H):
                for i0, i2 in grid(B=1, M=1):
                    call_extern("Posit{bits}es{es}QuireMatmul", A_flat, offsetA, K, B_flat, offsetB, i3, N, C_flat, offsetC)
"""

import tvm
from tvm import tir
from tvm.tir.functor import PyStmtExprMutator
import re


_POSIT_DTYPE_RE = re.compile(r"^custom\[posites(\d+)\](\d+)(x\d+)?$")


def get_quire_extern_symbol_from_dtype(dtype, with_offset):
    """Return extern symbol name for a custom posit dtype.

    Parameters
    ----------
    dtype : Union[str, tvm.DataType]
        Buffer dtype to inspect.
    with_offset : bool
        True for the 9-arg function (A/B offsets), False for 7-arg Elem variant.

    Returns
    -------
    Optional[str]
        Symbol name like "Posit12es1QuireMatmulElem" or None if dtype is not custom posit.
    """
    dtype_str = str(dtype)
    match = _POSIT_DTYPE_RE.fullmatch(dtype_str)
    if match is None:
        return None

    es = int(match.group(1))
    bits = int(match.group(2))
    suffix = "QuireMatmul" if with_offset else "QuireMatmulElem"
    return f"Posit{bits}es{es}{suffix}"


def transform_matmul_to_quire_elem(func: tir.PrimFunc) -> tir.PrimFunc:
    """
    TIR pass that transforms standard matmul loops into QuireMatmul calls.
    Supports 2D, 3D, and 4D matmul patterns.
    """
    
    @tir.functor.mutator
    class MatmulTransformer(PyStmtExprMutator):
        def __init__(self):
            super().__init__()
            self.transformed = False
        
        def visit_for_(self, node):
            """Check for matmul pattern and transform"""
            # Try to match different matmul patterns
            
            # All matmul patterns start with at least 2 nested For loops
            if not isinstance(node.body, tir.For):
                return super().visit_for_(node)
            
            loop1 = node  # Outermost loop
            loop2 = node.body  # Second loop
            
            if not isinstance(loop2.body, tir.For):
                return super().visit_for_(node)
            
            loop3 = loop2.body  # Third loop
            
            # Pattern 1: 2D matmul (3 nested loops)
            # for i1 (parallel) -> for i0 (unroll) -> for k (serial) -> BlockRealize
            if isinstance(loop3.body, tir.BlockRealize):
                block_realize = loop3.body
                block = block_realize.block
                
                # Check if this is a 2D matmul block (3 iter vars)
                if self.is_matmul_block(block, expected_dims=3):
                    return self.transform_2d_matmul(loop1, loop2, loop3, block_realize, block)
            
            # Pattern 2 & 3: 3D or 4D matmul (4+ nested loops)
            # for outer -> for middle -> for i1 -> for k -> BlockRealize (3D)
            # for outer -> for middle -> for i0 -> for i2 -> for k -> BlockRealize (4D)
            if isinstance(loop3.body, tir.For):
                loop4 = loop3.body  # Fourth loop
                
                # Check for 3D (4 loops total)
                if isinstance(loop4.body, tir.BlockRealize):
                    block_realize = loop4.body
                    block = block_realize.block
                    
                    # Check if this is a 3D matmul block (4 iter vars)
                    if self.is_matmul_block(block, expected_dims=4):
                        return self.transform_3d_matmul(loop1, loop2, loop3, loop4, block_realize, block)
                
                # Check for 4D (5 loops total)
                if isinstance(loop4.body, tir.For):
                    loop5 = loop4.body  # Fifth loop
                    
                    if isinstance(loop5.body, tir.BlockRealize):
                        block_realize = loop5.body
                        block = block_realize.block
                        
                        # Check if this is a 4D matmul block (5 iter vars)
                        if self.is_matmul_block(block, expected_dims=5):
                            return self.transform_4d_matmul(loop1, loop2, loop3, loop4, loop5, block_realize, block)
            
            # Not a matmul pattern, continue normal traversal
            return super().visit_for_(node)
        
        def is_matmul_block(self, block, expected_dims):
            """
            Check if block is a matmul reduction.
            expected_dims: 3 for 2D (i0, i1, k), 4 for 3D (i0, i1, i2, k), 5 for 4D (i0, i1, i2, i3, k)
            """
            # Must have expected number of iter vars
            if len(block.iter_vars) != expected_dims:
                return False
            
            # Must have exactly one reduction axis (the last one)
            reduction_count = sum(
                1 for iv in block.iter_vars 
                if iv.iter_type == tir.IterVar.CommReduce
            )
            if reduction_count != 1:
                return False
            
            # The last iter var must be the reduction axis
            if block.iter_vars[-1].iter_type != tir.IterVar.CommReduce:
                return False
            
            # Check for multiply-add pattern in BufferStore
            if not isinstance(block.body, tir.BufferStore):
                return False
            
            store = block.body
            if isinstance(store.value, tir.Add):
                add = store.value
                if isinstance(add.b, tir.Mul):
                    # Additional check: ensure reads contain BufferLoad (not Call nodes)
                    # Matmul blocks should have buffer reads, not extern calls
                    if len(block.reads) >= 2:
                        # Check that we're reading from actual buffers
                        for read_region in block.reads:
                            if not hasattr(read_region, 'buffer'):
                                return False
                        return True
            
            return False
        
        def transform_2d_matmul(self, i1_loop, i0_loop, k_loop, block_realize, block):
            """Transform 2D matmul loops to QuireMatmulElem call"""
            # Extract dimensions from loops
            M = i0_loop.extent
            K = k_loop.extent
            N = i1_loop.extent
            
            i0_var = i0_loop.loop_var
            i1_var = i1_loop.loop_var
            
            # Get buffers
            C_buffer = block.writes[0].buffer
            A_buffer = block.reads[0].buffer
            B_buffer = block.reads[1].buffer

            extern_symbol = get_quire_extern_symbol_from_dtype(C_buffer.dtype, with_offset=True)
            if extern_symbol is None:
                return i1_loop
            
            # Create extern call
            offset_C = i0_var * N + i1_var
            offset_A = i0_var * K
            offset_B = tir.const(0, "int64")
            extern_call = tir.Evaluate(
                tir.call_extern(
                    "int32",
                    extern_symbol,
                    A_buffer.data,
                    offset_A,
                    K,
                    B_buffer.data,
                    offset_B,
                    i1_var,
                    N,
                    C_buffer.data,
                    offset_C
                )
            )
            
            # Create new i0 loop with extern call (no k loop)
            new_i0_loop = tir.For(
                i0_loop.loop_var,
                i0_loop.min,
                i0_loop.extent,
                i0_loop.kind,
                extern_call,
                i0_loop.thread_binding,
                i0_loop.annotations
            )
            
            # Create new i1 loop
            new_i1_loop = tir.For(
                i1_loop.loop_var,
                i1_loop.min,
                i1_loop.extent,
                i1_loop.kind,
                new_i0_loop,
                i1_loop.thread_binding,
                i1_loop.annotations
            )
            
            self.transformed = True
            return new_i1_loop
        
        def transform_3d_matmul(self, i2_loop, i0_loop, i1_loop, k_loop, block_realize, block):
            """
            Transform 3D matmul: (batch, M, K) @ (K, N) -> (batch, M, N)
            Pattern: for i2(N) -> for i0(batch) -> for i1(M) -> for k(K) -> block
            
            Transform to:
                for i2(N) -> for i0(batch) -> for i1(M) -> QuireMatmulElemWithOffset call
            """
            # Extract loop variables and extents
            i2_var = i2_loop.loop_var  # N dimension (parallel)
            i0_var = i0_loop.loop_var  # batch dimension (unroll)
            i1_var = i1_loop.loop_var  # M dimension
            
            N = i2_loop.extent
            batch = i0_loop.extent
            M_extent = i1_loop.extent
            K_extent = k_loop.extent
            
            # Get buffers from block
            C_buffer = block.writes[0].buffer
            A_buffer = block.reads[0].buffer
            B_buffer = block.reads[1].buffer

            extern_symbol = get_quire_extern_symbol_from_dtype(C_buffer.dtype, with_offset=True)
            if extern_symbol is None:
                return i2_loop
            
            # Calculate offsets for flattened access
            # C[i0, i1, i2]: flattened offset = i0 * M * N + i1 * N + i2
            offset_C = i0_var * M_extent * N + i1_var * N + i2_var
            
            # A[i0, i1, k]: offset to A[i0, i1, 0] = i0 * M * K + i1 * K
            offset_A = i0_var * M_extent * K_extent + i1_var * K_extent
            
            # B offset is 0 for 3D matmul (B is still 2D)
            offset_B = tir.const(0, "int64")
            
            # Create QuireMatmulElemWithOffset extern call
            extern_call = tir.Evaluate(
                tir.call_extern(
                    "int32",
                    extern_symbol,
                    A_buffer.data,
                    offset_A,
                    K_extent,
                    B_buffer.data,
                    offset_B,
                    i2_var,
                    N,
                    C_buffer.data,
                    offset_C
                )
            )
            
            # Create new i1 loop (M dimension) - replace k_loop with extern call
            new_i1_loop = tir.For(
                i1_loop.loop_var,
                i1_loop.min,
                i1_loop.extent,
                i1_loop.kind,
                extern_call,
                i1_loop.thread_binding,
                i1_loop.annotations
            )
            
            # Create new i0 loop (batch)
            new_i0_loop = tir.For(
                i0_loop.loop_var,
                i0_loop.min,
                i0_loop.extent,
                i0_loop.kind,
                new_i1_loop,
                i0_loop.thread_binding,
                i0_loop.annotations
            )
            
            # Create new i2 loop (N)
            new_i2_loop = tir.For(
                i2_loop.loop_var,
                i2_loop.min,
                i2_loop.extent,
                i2_loop.kind,
                new_i0_loop,
                i2_loop.thread_binding,
                i2_loop.annotations
            )
            
            self.transformed = True
            return new_i2_loop
        
        def transform_4d_matmul(self, i3_loop, i1_loop, i0_loop, i2_loop, k_loop, block_realize, block):
            """
            Transform 4D matmul: (B, H, M, K) @ (B, H, K, N) -> (B, H, M, N)
            Pattern: for i3(N) -> for i1(H) -> for i0(B) -> for i2(M) -> for k(K) -> block
            
            Transform to:
                for i3(N) -> for i1(H) -> for i0(B) -> for i2(M) -> QuireMatmulElemWithOffset call
            """
            # Extract loop variables and extents
            i3_var = i3_loop.loop_var  # N dimension (parallel)
            i1_var = i1_loop.loop_var  # H (heads) dimension (unroll)
            i0_var = i0_loop.loop_var  # B (batch) dimension
            i2_var = i2_loop.loop_var  # M dimension
            
            N = i3_loop.extent
            H = i1_loop.extent
            B_extent = i0_loop.extent
            M_extent = i2_loop.extent
            K_extent = k_loop.extent
            
            # Get buffers from block
            C_buffer = block.writes[0].buffer
            A_buffer = block.reads[0].buffer
            B_buffer = block.reads[1].buffer

            extern_symbol = get_quire_extern_symbol_from_dtype(C_buffer.dtype, with_offset=True)
            if extern_symbol is None:
                return i3_loop
            
            # Calculate offsets for flattened access
            # C[i0, i1, i2, i3]: offset = i0*H*M*N + i1*M*N + i2*N + i3
            offset_C = i0_var * H * M_extent * N + i1_var * M_extent * N + i2_var * N + i3_var
            
            # A[i0, i1, i2, k]: offset to A[i0, i1, i2, 0] = i0*H*M*K + i1*M*K + i2*K
            offset_A = i0_var * H * M_extent * K_extent + i1_var * M_extent * K_extent + i2_var * K_extent
            
            # B[i0, i1, k, i3]: offset to B[i0, i1, 0, :] = i0*H*K*N + i1*K*N
            offset_B = i0_var * H * K_extent * N + i1_var * K_extent * N
            
            # Create QuireMatmulElemWithOffset extern call
            extern_call = tir.Evaluate(
                tir.call_extern(
                    "int32",
                    extern_symbol,
                    A_buffer.data,
                    offset_A,
                    K_extent,
                    B_buffer.data,
                    offset_B,
                    i3_var,
                    N,
                    C_buffer.data,
                    offset_C
                )
            )
            
            # Create new i2 loop (M dimension) - replace k_loop with extern call
            new_i2_loop = tir.For(
                i2_loop.loop_var,
                i2_loop.min,
                i2_loop.extent,
                i2_loop.kind,
                extern_call,
                i2_loop.thread_binding,
                i2_loop.annotations
            )
            
            # Create new i0 loop (B/batch dimension)
            new_i0_loop = tir.For(
                i0_loop.loop_var,
                i0_loop.min,
                i0_loop.extent,
                i0_loop.kind,
                new_i2_loop,
                i0_loop.thread_binding,
                i0_loop.annotations
            )
            
            # Create new i1 loop (H/heads)
            new_i1_loop = tir.For(
                i1_loop.loop_var,
                i1_loop.min,
                i1_loop.extent,
                i1_loop.kind,
                new_i0_loop,
                i1_loop.thread_binding,
                i1_loop.annotations
            )
            
            # Create new i3 loop (N)
            new_i3_loop = tir.For(
                i3_loop.loop_var,
                i3_loop.min,
                i3_loop.extent,
                i3_loop.kind,
                new_i1_loop,
                i3_loop.thread_binding,
                i3_loop.annotations
            )
            
            self.transformed = True
            return new_i3_loop
    
    # Apply transformation
    transformer = MatmulTransformer()
    
    # Handle root block if present
    body = func.body
    if isinstance(body, tir.BlockRealize) and body.block.name_hint == "root":
        # Visit the body of the root block
        new_body = transformer.visit_stmt(body.block.body)
        # Reconstruct root block with transformed body
        new_root_block = tir.Block(
            iter_vars=body.block.iter_vars,
            reads=body.block.reads,
            writes=body.block.writes,
            name_hint=body.block.name_hint,
            body=new_body,
            init=body.block.init,
            alloc_buffers=body.block.alloc_buffers,
            match_buffers=body.block.match_buffers,
            annotations=body.block.annotations
        )
        new_body = tir.BlockRealize(
            iter_values=body.iter_values,
            predicate=body.predicate,
            block=new_root_block
        )
    else:
        new_body = transformer.visit_stmt(body)
    
    if transformer.transformed:
        # Create new function with transformed body
        new_func = tir.PrimFunc(
            params=func.params,
            body=new_body,
            ret_type=func.ret_type,
            buffer_map=func.buffer_map,
            attrs=func.attrs
        )
        # Add tir.noalias attribute
        return new_func.with_attr("tir.noalias", True)
    else:
        return func


@tvm.tir.transform.prim_func_pass(opt_level=0)
class InjectQuireMatmulElem:
    """
    TVM pass to transform matmul operations to use QuireMatmulElem extern calls
    
    Usage:
        mod = IRModule({"matmul": matmul_func})
        transformed_mod = InjectQuireMatmulElem()(mod)
    """
    
    def __init__(self):
        pass
    
    def transform_function(self, func, mod, ctx):
        return transform_matmul_to_quire_elem(func)


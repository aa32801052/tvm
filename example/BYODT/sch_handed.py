from tvm import IRModule, tir
# Helper function to retrieve all blocks in a TIR function
def get_all_blocks(sch, func):
    blocks = []
    # Define a visitor to collect all blocks
    def visit_block(stmt):
        if isinstance(stmt, tir.Block):
            blocks.append(stmt)
    tir.stmt_functor.post_order_visit(func.body, visit_block)
    return blocks

# Function to apply automatic optimization to a given TIR function
def auto_optimize_func(sch, func, tile_sizes=(8, 16, 32, 16), use_vectorize=False):
    def custom_sort_key(item):
        index, value = item
        if isinstance(value, str):
            return (0, value)
        elif isinstance(value, int):
            return (1, -value) 
        return (2, index)

    # Collect all block names in the function
    blocks = get_all_blocks(sch, func)
    # Apply scheduling transformations to each block
    for block in blocks:
        block_name = block.name_hint
        # Get the target block
        blockRV = sch.get_block(block_name)
        # Get the loops of the block
        loops = sch.get_loops(blockRV)
        if len(loops) == 0:
            continue

        iter_vars = []
        for index, iter_var in enumerate(block.iter_vars):
            if iter_var.iter_type == iter_var.CommReduce:
                continue
            if type(iter_var.dom.extent) is tir.expr.IntImm:
                value = iter_var.dom.extent.value
            else:
                value = 'none'
            iter_vars.append(value)
        if len(iter_vars) > 0:
            iter_vars = [(index, value) for index, value in enumerate(iter_vars)]
            # print(iter_vars)
            sorted_list = sorted(iter_vars, key=custom_sort_key)
            # print("block:", block_name)
            # print("  #loops:", len(loops))
            # print("  iter_vars:", iter_vars)
            # print("  sorted_list:", sorted_list)
            # Check if there are enough loops to tile
            #if 'matmul' in block_name:

            reorder_list = []
            for i in range(len(sorted_list)):
                reorder_list.append(loops[sorted_list[i][0]])
            sch.reorder(*reorder_list)

            for i in range(0, len(sorted_list)):
                sch.parallel(loops[sorted_list[i][0]])
                sorted_list[i] = (-1, sorted_list[i][1])
                break
            
            # Use vectorize for float types
            if use_vectorize:
                for i in range(0, len(sorted_list)):
                    if (
                        sorted_list[i][0] != -1
                        and isinstance(sorted_list[i][1], int)
                        and sorted_list[i][1] > 1
                        and sorted_list[i][1] <= 256
                    ):
                        sch.vectorize(loops[sorted_list[i][0]])
                        sorted_list[i] = (-1, sorted_list[i][1])
                        break

            for i in range(0, len(sorted_list)):
                if (
                    sorted_list[i][0] != -1
                    and isinstance(sorted_list[i][1], int)
                    and sorted_list[i][1] <= 128
                ):
                    sch.unroll(loops[sorted_list[i][0]])
                    sorted_list[i] = (-1, sorted_list[i][1])
                    break


    #print(sch.mod.script())
    # Return the modified function from the schedule
    return sch.mod["main"]

# Function to optimize all TIR functions in the IR module
def optimize_ir_module(ir_module, use_vectorize=False):
    """ Optimize all TIR functions in the IR module.
    Args:
        ir_module: Input IR module
        use_vectorize: Whether to use vectorization
    """
    optimized_module = IRModule()

    # Iterate over each function in the IR module
    for name, func in ir_module.functions.items():
        if isinstance(func, tir.PrimFunc):  # Only apply to TIR functions
            # Create a schedule and apply the optimizations
            sch = tir.Schedule(func)
            optimized_func = auto_optimize_func(sch, func, use_vectorize=use_vectorize)
            optimized_module[name] = optimized_func
        else:
            optimized_module[name] = func

    return optimized_module
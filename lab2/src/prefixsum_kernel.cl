void kernel sum_array(global float const *blocks_sums,
                      global float *output) {
    int block_i = get_group_id(0) - 1;
    if (block_i >= 0) {
        output[get_global_id(0)] += blocks_sums[block_i];
    }
}

inline void swap(local float **a, local float **b) {
    local float * tmp = *a;
    *a = *b;
    *b = tmp;
}

// Hills-Steele scan
void kernel sum_blocks(global float const *array,
                       global float *block_sum,
                       local float *block_sums,
                       local float *aux,
                       global float *output) {

    int array_i = get_global_id(0);
    int index = get_local_id(0);
    int block_sz = get_local_size(0);
    int block_i = get_group_id(0);

    block_sums[index] = array[array_i];

    for(int shift = 1; shift < block_sz; shift += shift) {

        if (index < shift) {
            aux[index] = block_sums[index];
        } else {
            aux[index] = block_sums[index] + block_sums[index - shift];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        swap(&block_sums, &aux);
    }

    output[array_i] = block_sums[index];

    if (index + 1 == block_sz) {
        block_sum[block_i] = output[array_i];
    }
}
__kernel void convolute(__global const float* a, unsigned n,
                            __global const float* b, unsigned m,
                            __global float* res) {
    int m2 = m / 2;
    size_t row = get_global_id(0);
    size_t col = get_global_id(1);

    if (row >= n || col >= n) {
        return;
    }

    float sum = 0;
    for (int i = -m2; i <= m2; ++i) {
        for (int j = -m2; j <= m2; ++j) {
            if (row + i >= 0 && row + i < n && col + j >= 0 && col + j < n) {
                sum += b[(i + m2) * m + j + m2] * a[(row + i) * n + col + j];
            }
        }
    }

    res[row * n + col] = sum;
}
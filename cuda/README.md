# Simple CUDA Vector Add

This is the smallest useful version of vector add:

- `vector_add.cu` has one CUDA kernel, `vecAdd`
- `main()` creates three arrays of size `500`
- `A` is filled with `1, 3, 5, ...`
- `B` is filled with `2, 4, 6, ...`
- the arrays are copied to the GPU
- the kernel runs
- the result is copied back and printed

## Run in Colab

Open [colab_vector_add_demo.ipynb](/Users/rahulbir/Desktop/kernel-pg/cuda/colab_vector_add_demo.ipynb) in Colab, switch the runtime to GPU, and run the cells.

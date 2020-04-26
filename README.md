# cuMultigrid.jl
Multigrid solver based on Julia CUDA (2D Poisson problem demo)

Currently it only supports solution grids with side lengths (2^n)-1, e.g. 1023x1023, 2047x2047. It can be extended to support all grid sizes by replacing the restriction and prolongation functions.


Notes on improvement:

- Make use of shared memory

- Small grids (smaller than 512x512 or 1024x1024) can be solved in a single block without much difficulty, it reduces the time for CPU-GPU synchronization

- Very coarse grids (e.g. 4x4) can be solved directly, instead of having to "recurse" until 1x1 grid

- Reduce CPU-GPU synchronization in Red-black Gauss Seidel step, or simply replace it with another smoother


Performance:

As expected for a correct implementation, the time for moderate grid sizes (511x511 to 2047x2047) increases linearly. However, the time jumps significantly as the grid size reaches 4095x4095. 

Using norm instead of findmax for error calculation may improve performance. The implementations of these functions in the Julia GPU library are being updated from time to time.

In preliminary trials on K80, this program outperforms both a CUDA/C equivalent (mentioned next section) and NVIDIA's AMGX. The results are very (way too) surprising and are indicated at the bottom of this README. Nonetheless both the Julia and CUDA program are exponentially faster than AMGX due to different asymptotic time curves (linear vs exponential). 


Notes on CUDA/C predecessor:

This program was migrated from a CUDA/C program that I wrote in 2018. The CUDA/C program makes use of a single block to solve problems up to 1024x1024 in size. It was very performant, faster than a Conjugate Gradient program using CuBLAS and CuSparse, and much faster than the AMGX library (about 19.3x faster). 


Optimizations used by both programs include:

- Elimination of recursion, though the effect on the Julia program should be limited because the recursion is performed on the CPU

- Reusing memory, some memory spaces for the V-cycle can be reused in different stages, it saves about 50% in memory compared to allocating a space for each array in the V-cycle algorithm


Preliminary results:

On a K80, the time reported to solve a 1023x1023 2D Poisson problem was 0.07s.

By comparing benchmarks on K80, it is expected to be 363 times faster than NVIDIA's AMGX on moderate grid sizes, and approximately 19 times faster than the CUDA equivalent. These results shall be further verified and the cause of any difference in performance between the Julia and CUDA program shall be investigated. However, it should be emphasized that Julia, despite being a higher-level, GC language, can run faster than CUDA https://devblogs.nvidia.com/gpu-computing-julia-programming-language/.

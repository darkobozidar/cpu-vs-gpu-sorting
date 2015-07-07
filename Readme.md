## Info

A comparison study between sequential sorting algorithms implemented in C++ and parallel sorting algorithms implemented in CUDA as part of the master's thesis.
We implemented seven algorithms: bitonic sort, multistep bitonic sort, adaptive bitonic sort, merge
sort, quicksort, radix sort and sample sort.
Sequential algorithms were implemented on a CPU using C++, whereas parallel algorithms were implemented on a GPU using CUDA platform.
We improved the above mentioned implementations and adopted them to be able to sort input sequences of arbitrary length.
We compared algorithms on six different input distributions, which consist of 32-bit numbers, 32-bit
key-value pairs, 64-bit numbers and 64-bit key-value pairs.
The results show that radix sort is the fastest sequential sorting algorithm, whereas radix sort and merge sort are the fastest parallel sorting algorithms (depending on the input distribution).
With parallel implementations we achieved speedups of up to 157-times in comparison to sequential implementations.

- **Author**: Darko Božidar
- **Mentor**: Tomaž Dobravec, Ph.D.

## Downloads

- **Results**: https://drive.google.com/file/d/0B7uLFueU4vLfcFBmNWNoODc5TE0/view?usp=sharing
- **Master's thesis (slovenian version)**: https://drive.google.com/file/d/0B7uLFueU4vLfM3IwazZ0VFB5Q0E/view?usp=sharing
- **Paper**: https://drive.google.com/file/d/0B7uLFueU4vLfcjJfZFh3TlIxMFE/view?usp=sharing

**Note**: in case of any broken links please contact me on *darko.bozidar@gmail.com*.

## Dependencies

-  CUDPP 2.2

## Sorting algorithms

#### Sequential algorithms:

- Bitonic sort: [1], [2]
- Adaptive bitonic sort: [4]
- Merge sort: [5]
- Quicksort: [5]
- Radix sort: [5]
- Sample sort: [5], [17]

#### Parallel algorithms:

- Bitonic sort: [1], [2]
- Multistep bitonic sort: [2]
- Adaptive bitonic sort: [3], [4]
- Merge sort: [6], [7], [8]
- Quicksort: [9], [10], [11], [12], [13], [14], [15]
- Radix sort: [6], [15], [16]
- Sample sort: [2], [15], [18]


## References


[1] K. E. Batcher. Sorting networks and their applications. In Proceedings of the April 30-May 2, 1968, Spring Joint Computer Conference, AFIPS '68 (Spring), pages 307-314, New York, NY, USA, 1968. ACM.

[2] H. Peters, O. Schulz-Hildebrandt, and N. Luttenberger. Fast in-place, comparison-based sorting with CUDA: A study with bitonic sort. Concurr. Comput. : Pract. Exper., 23(7):681-693, May 2011.

[3] H. Peters, O. Schulz-Hildebrandt, and N. Luttenberger. A novel sorting algorithm for many-core architectures based on adaptive bitonic sort. In 26th IEEE International Parallel and Distributed Processing Symposium, IPDPS 2012, Shanghai, China, May 21-25, 2012, pages 227-237, 2012.

[4] G. Bilardi and A. Nicolau. Adaptive bitonic sorting: An optimal parallel algorithm for shared memory machines. Technical report, Ithaca, NY, USA, 1986.

[5] T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein. Introduction to Algorithms, Third Edition. The MIT Press, 3rd edition, 2009.

[6] N. Satish, M. Harris, and M. Garland. Designing efficient sorting algorithms for manycore GPUs. In Proceedings of the 2009 IEEE International Symposium on Parallel&Distributed Processing, IPDPS '09, pages 1-10, Washington, DC, USA, 2009. IEEE Computer Society.

[7] T. Hagerup and C. Rub. Optimal merging and sorting on the erew pram. Inf. Process. Lett., 33(4):181-185, Dec. 1989.

[8] D. Z. Chen. Efficient parallel binary search on sorted arrays, with applications. IEEE Trans. Parallel Distrib. Syst., 6(4):440-445, Apr. 1995.

[9] D. Cederman and P. Tsigas. GPU-quicksort: A practical quicksort algorithm for graphics processors. J. Exp. Algorithmics, 14:4:1.4-4:1.24, Jan. 2010.

[10] R. C. Singleton. Algorithm 347: An efficient algorithm for sorting with minimal storage [m1]. Commun. ACM, 12(3):185-186, Mar. 1969.

[11] R. Sedgewick. Implementing quicksort programs. Commun. ACM, 21(10):847-857, Oct. 1978.

[12] M. Harris. Optimizing Parallel Reduction in CUDA. Technical report, nVidia, 2008.

[13] D. B. Kirk and W.-m. W. Hwu. Programming Massively Parallel Processors: A Hands-on Approach. Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 2 edition, 2013.

[14] S. Sengupta, M. Harris, and M. Garland. Efficient parallel scan algorithms for GPUs. Technical Report NVR-2008-003, NVIDIA Corporation, Dec. 2008.

[15] Cudpp: CUDA data parallel primitives library. https://github.com/cudpp/cudpp/, 2015.

[16] M. Harris and M. Garland. Chapter 3 - Optimizing Parallel Prefix Operations for the Fermi Architecture. In W.-m. W. Hwu, editor, GPU Computing Gems Jade Edition, Applications of GPU Computing Series, pages 29 - 38. Morgan Kaufmann, Boston, 2012.

[17] N. Leischner, V. Osipov, and P. Sanders. GPU sample sort. In 24th IEEE International Symposium on Parallel and Distributed Processing, IPDPS 2010, Atlanta, Georgia, USA, 19-23 April 2010 - Conference Proceedings, pages 1-10, April 2010.

[18] F. Dehne and H. Zaboli. Deterministic sample sort for GPUs. CoRR, abs/1002.4464, 2010.

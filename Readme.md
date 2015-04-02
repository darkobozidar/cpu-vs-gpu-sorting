All algorithms can sort keys and key-value pairs.

Dependencies:
- CUDPP 2.2


Results: http://x.k00.fr/lr3rd

Current version of master's thesis (slovenian version): http://x.k00.fr/na792


Sequential algorithms:
- Bitonic sort: [1], [2]
- Adaptive bitonic sort: [4]
- Merge sort: [5]
- Quicksort: [5]
- Radix sort: [5]
- Sample sort: [5], [17]

Parallel algorithms:
- Bitonic sort: [1], [2]
- Multistep bitonic sort: [2]
- Adaptive bitonic sort: [3], [4]
- Merge sort: [6], [7], [8]
- Quicksort: [9], [10], [11], [12], [13], [14], [15]
- Radix sort: [6], [15], [16]
- Sample sort: [2], [15], [18]


Literature:

[1] K. E. Batcher, "Sorting networks and their applications",
    in Proceedings of the April 30-May 2, 1968, Spring Joint Computer Conference,
    AFIPS '68 (Spring), (New York, NY, USA), pp. 307-314, ACM, 1968.

(NOTE: there are two very similar versions of this paper - 2009 and 2011)
[2] H. Peters, O. Schulz-Hildebrandt, and N. Luttenberger,
    "Fast in-place, comparison-based sorting with cuda: A study with bitonic sort",
    Concurr. Comput. : Pract. Exper., vol. 23, pp. 681-693, May 2011.

[3] H. Peters, O. Schulz-Hildebrandt, and N. Luttenberger,
    "A novel sorting algorithm for many-core architectures based on adaptive bitonic sort",
    in 26th IEEE International Parallel and Distributed Processing Symposium,
    IPDPS 2012, Shanghai, China, May 21-25, 2012, pp. 227-237, 2012.

[4] G. Bilardi and A. Nicolau, "Adaptive bitonic sorting: An optimal parallel
    algorithm for shared memory machines", tech. rep., Ithaca, NY, USA, 1986.

[5] T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein, Introduction
    to Algorithms, Third Edition. The MIT Press, 3rd ed., 2009.

[6] N. Satish, M. Harris, and M. Garland, "Designing efficient sorting algorithms
    for manycore gpus", in Proceedings of the 2009 IEEE International
    Symposium on Parallel&Distributed Processing, IPDPS '09,
    (Washington, DC, USA), pp. 1-10, IEEE Computer Society, 2009.

[7] T. Hagerup and C. Rub, "Optimal merging and sorting on the erew pram",
    Inf. Process. Lett., vol. 33, pp. 181-185, Dec. 1989.

[8] D. Z. Chen, "Efficient parallel binary search on sorted arrays, with applications",
    IEEE Trans. Parallel Distrib. Syst., vol. 6, pp. 440-445, Apr. 1995.

(NOTE: there are three very similar versions of this paper - from 2008 to 2010)
[9] D. Cederman and P. Tsigas, "Gpu-quicksort: A practical quicksort algorithm
    for graphics processors", J. Exp. Algorithmics, vol. 14, pp. 4:1.4-4:1.24, Jan. 2010.

[10] R. C. Singleton, "Algorithm 347: An efficient algorithm for sorting with minimal storage [m1]",
     Commun. ACM, vol. 12, pp. 185-186, Mar. 1969.

[11] R. Sedgewick, "Implementing quicksort programs",
     Commun. ACM, vol. 21, pp. 847-857, Oct. 1978.

[12] M. Harris, "Optimizing Parallel Reduction in CUDA", tech. rep., nVidia, 2008.

[13] D. B. Kirk and W.-m. W. Hwu, Programming Massively Parallel Processors: A Hands-on Approach.
     San Francisco, CA, USA: Morgan Kaufmann Publishers Inc., 2 ed., 2013.

[14] S. Sengupta, M. Harris, and M. Garland, "Efficient parallel scan algorithms for GPUs,"
     Tech. Rep. NVR-2008-003, NVIDIA Corporation, Dec. 2008.

[15] "Cudpp: Cuda data parallel primitives library."
     https://github.com/cudpp/cudpp/, Feb 2015.

[16] M. Harris and M. Garland, "Chapter 3 - optimizing parallel prefix operations
     for the fermi architecture", in fGPUg Computing Gems Jade Edition (W.-m. W. Hwu, ed.),
	 Applications of GPU Computing Series, pp. 29 - 38, Boston: Morgan Kaufmann, 2012.

[17] N. Leischner, V. Osipov, and P. Sanders, "Gpu sample sort.",
     in IPDPS, pp. 1-10, IEEE, 2010.

[18] F. Dehne and H. Zaboli, "Deterministic sample sort for gpus",
     CoRR, vol. abs/1002.4464, 2010.

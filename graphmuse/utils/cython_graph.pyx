# cython: language_level=3
# distutils: language=c++
# distutils: extra_compile_args = -fopenmp -std=c++11
# distutils: extra_link_args = -fopenmp
cimport cython
from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
from libc.math cimport fabs
from libcpp.vector cimport vector
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cimport graphmuse.utils.cython_utils as cutils
import graphmuse.utils.cython_utils as cutils


cdef extern from "stdlib.h":
    int RAND_MAX


DTYPE = np.int
cdef double INF = np.inf
cdef float SMALL = 1e-6


cdef int isclose(float a, float b) nogil:
    if (fabs(a - b) <= SMALL):
        return 1
    else:
        return 0

#
# cdef class GraphFromEdges:
#     cdef vector[float] onset_beat
#     cdef vector[float] duration_beat
#     cdef vector[float] pitch
#     cdef vector[vector[int]] adj
#
#     def __cinit__(self, np.ndarray[float, ndim = 1, mode = 'c'] onset_beat,
#         np.ndarray[float, ndim = 1, mode = 'c'] duration_beat,
#         np.ndarray[float, ndim = 1, mode = 'c'] pitch,
#         int num_proc
#         ):
#         cutils.npy2vec_float(onset_beat, self.onset_beat)
#         cutils.npy2vec_float(duration_beat, self.duration_beat)
#         cutils.npy2vec_float(pitch, self.pitch)
#         self.src = vector[int]()
#         self.dst = vector[int]()
#         self.num_proc = num_proc
#
#     cdef void create_edges(self, int N, int idx, vector[int] src, vector[int] dst) nogil:
#         cdef int j = 0
#
#         for j in range(N):
#             if isclose(self.onset_beat[j], self.onset_beat[idx]) and self.pitch[j] != self.pitch[idx]:
#                 src.push_back(idx)
#                 dst.push_back(j)
#             if isclose(self.onset_beat[j], self.onset_beat[idx] + self.duration_beat[idx]):
#                 src.push_back(idx)
#                 dst.push_back(j)
#             if (self.onset_beat[idx] < self.onset_beat[j]) and (self.onset_beat[idx] + self.duration_beat[idx] > self.onset_beat[j]):
#                 src.push_back(idx)
#                 dst.push_back(j)
#
#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     def process(self):
#         cdef int p = 0
#         cdef int i = 0
#         cdef int j = 0
#         cdef int N = len(self.onset_beat)
#         cdef float et
#         cdef int M
#         cdef int T = 0
#         cdef int R = 0
#         cdef float tmp_min
#         cdef int num_proc = self.num_proc
#         cdef np.ndarray[float, ndim=2] end_times = np.add(self.onset_beat, self.duration_beat)
#         cdef np.ndarray[float, ndim=1] uniques = np.sort(np.unique(end_times))[:-1]
#         cdef np.ndarray[int, ndim=1] tmp
#         cdef np.ndarray[int, ndim=2] edges
#         cdef int workload = N // num_proc
#         cdef int wbegin
#         cdef int wend
#         cdef vector[vector[int]] src_buffers
#         cdef vector[vector[int]] dst_buffers
#         cdef vector[float].iterator it
#         cdef vector[float].iterator it_end
#         cdef vector[float].iterator find
#
#         with nogil, parallel(num_threads=num_proc):
#             for p in prange(num_proc, schedule='dynamic'):
#                 wbegin = workload*p
#                 wend = workload*(p+1)
#                 for i in range(wbegin, wend, 1):
#                     self.create_edges(N, i, src_buffers[p], dst_buffers[p])
#
#         for i in range(num_proc):
#             M = src_buffers[i].size()
#             for j in range(M):
#                 self.src.push_back(src_buffers[i][j])
#                 self.dst.push_back(dst_buffers[i][j])
#
#         i = 0
#         j = 0
#         p = 0
#         M = len(uniques)
#         # Try to make this parallelized.
#         for p in range(M):
#             et = uniques[p]
#             find = find(it, it_end, et)
#             if find != it_end:
#                 scr = np.where(end_times == et)[0]
#                 diffs = np.diff(self.onset_beat, et)
#                 tmp = np.where(diffs > 0, diffs, INF)
#                 tmp_min = tmp.min()[0]
#                 dst = np.where(tmp == tmp_min)[0]
#                 T = scr.shape[0]
#                 R = dst.shape[0]
#                 for i in range(T):
#                     for j in range(R):
#                         self.src.push_back(i)
#                         self.dst.push_back(j)
#         edges = np.ndarray([self.src, self.dst])
#         return edges
from cpython cimport array
import array

cdef class GraphFromAdj:
    cdef vector[float] onset_beat
    cdef vector[float] duration_beat
    cdef vector[float] pitch
    cdef vector[vector[int]] adj
    cdef int N
    cdef int num_proc

    def __cinit__(self, np.ndarray[float, ndim = 1, mode = 'c'] onset_beat,
        np.ndarray[float, ndim = 1, mode = 'c'] duration_beat,
        np.ndarray[float, ndim = 1, mode = 'c'] pitch,
        int num_proc
        ):
        cutils.npy2vec_float(onset_beat, self.onset_beat)
        cutils.npy2vec_float(duration_beat, self.duration_beat)
        cutils.npy2vec_float(pitch, self.pitch)
        self.N = len(onset_beat)
        self.num_proc = num_proc



    cdef void create_edges(self, int n, int idx, int* adj) nogil:
        cdef int j = 0
        cdef int* row = adj + n*idx

        for j in range(n):
            row[j] = 0
            if isclose(self.onset_beat[j], self.onset_beat[idx]) and self.pitch[j] != self.pitch[idx]:
                row[j] = 1
            if isclose(self.onset_beat[j], self.onset_beat[idx] + self.duration_beat[idx]):
                row[j] = 1
            if (self.onset_beat[idx] < self.onset_beat[j]) and (self.onset_beat[idx] + self.duration_beat[idx] > self.onset_beat[j]):
                row[j] = 1




    @cython.boundscheck(False)
    @cython.wraparound(False)
    def process(self):
        cdef int p = 0
        cdef int i = 0
        cdef int j = 0
        cdef int N = self.onset_beat.size()
        cdef float et
        cdef int M
        cdef int T = 0
        cdef int R = 0
        cdef float tmp_min
        cdef int num_proc = self.num_proc
        cdef np.ndarray[float, ndim=1] end_times = np.add(self.onset_beat, self.duration_beat, dtype=np.float32)
        cdef np.ndarray[float, ndim=1] uniques = np.sort(np.unique(end_times))[:-1]
        cdef np.ndarray[int, ndim=1] tmp
        cdef np.ndarray[int, ndim=2] edges
        cdef int workload = self.N // num_proc
        cdef int wbegin
        cdef int wend
        # cdef np.ndarray[int, ndim=2, mode='c'] adj = np.zeros((N, N), order="C")
        # cdef vector[float].iterator it
        # cdef vector[float].iterator it_end
        # cdef vector[float].iterator find
        cdef np.ndarray[float, ndim=1, mode='c'] ob = np.ndarray(self.onset_beat)
        cdef int* adj = <int *>PyMem_Malloc(N*N*sizeof(int))

        # cutils.npymatrix2vec_int(adj, self.adj)
        with nogil, parallel(num_threads=num_proc):
            for p in prange(num_proc, schedule='dynamic'):
                wbegin = workload*p
                wend = workload*(p+1)
                for i in range(wbegin, wend, 1):
                    self.create_edges(N, i, adj)

        cdef int[:, :] arr = <int[:N,:N]> adj
        adjacency = np.asarray(arr)
        src, dst = np.nonzero(adjacency)
        i = 0
        j = 0
        p = 0
        M = len(uniques)
        # Try to make this parallelized.
        for p in range(M):
            et = uniques[p]
            T = np.count_nonzero(np.isin(ob, et))
            if T > 0:
                scr = np.where(end_times == et)[0]
                diffs = np.diff(ob, et)
                tmp = np.where(diffs > 0, diffs, INF)
                tmp_min = tmp.min()[0]
                dst = np.where(tmp == tmp_min)[0]
                T = scr.shape[0]
                R = dst.shape[0]
                for i in range(T):
                    for j in range(R):
                        self.src.push_back(i)
                        self.dst.push_back(j)
        edges = np.ndarray([self.src, self.dst])
        PyMem_Free(adj)
        return edges


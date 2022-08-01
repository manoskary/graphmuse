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

DTYPE = np.int
cdef double INF = np.inf
cdef float SMALL = 1e-6


cdef int isclose(float a, float b):
    if (fabs(a - b) <= SMALL):
        return 1
    else:
        return 0


cdef class CreateG:

    cdef __cinit__(self, np.ndarray[float, ndim = 1, mode = 'c'] onset_beat,
        np.ndarray[float, ndim = 1, mode = 'c'] duration_beat,
        np.ndarray[float, ndim = 1, mode = 'c'] pitch,
        int num_proc
        ):
        self.onset_beat = onset_beat
        self.duration_beat = duration_beat
        self.pitch = pitch
        self.src = vector[int]
        self.dst = vector[int]
        self.num_proc = num_proc

    cdef void create_edges(self, int N, int idx) nogil:
        cdef int j = 0

        for j in range(N):
            if isclose(self.onset_beat[j], self.onset_beat[idx]) and self.pitch[j] != self.pitch[idx]:
                self.src.push_back(idx)
                self.dst.push_back(j)
            if isclose(self.onset_beat[j], self.onset_beat[idx] + self.duration_beat[idx]):
                self.src.push_back(idx)
                self.dst.push_back(j)
            if (self.onset_beat[idx] < self.onset_beat[j]) and (self.onset_beat[idx] + self.duration_beat[idx] > self.onset_beat[j]):
                self.src.push_back(idx)
                self.dst.push_back(j)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def process(self):
        cdef int p = 0
        cdef int i = 0
        cdef int j = 0
        cdef int N = len(self.onset_beat)
        cdef float et
        cdef int M
        cdef int T = 0
        cdef int R = 0
        cdef float tmp_min
        cdef int num_proc = self.num_proc
        cdef np.ndarray[float, ndim=2] end_times = np.add(self.onset_beat, self.duration_beat)
        cdef np.ndarray[float, ndim=1] uniques = np.sort(np.unique(end_times))[:-1]
        cdef np.ndarray[int, ndim=1] tmp
        cdef np.ndarray[int, ndim=2] edges


        with nogil, parallel(num_threads=num_proc):
            for p in prange(N, schedule='dynamic'):
                self.create_edges(N, p)

        p = 0
        M = len(uniques)
        # Try to make this parallelized.
        for p in range(M):
            et = uniques[p]
            if et not in self.onset_beat:
                scr = np.where(end_times == et)[0]
                diffs = self.onset_beat - et
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
        return edges


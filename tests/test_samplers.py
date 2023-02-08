import graphmuse.samplers as sam
import unittest


class TestSamplers(unittest.TestCase):

        def test_gmsamplersmodule(self):
            import numpy
            
            t = numpy.linspace(0,1,20)

            src = (9*(t**2)).astype(numpy.uint32)
            dst = (numpy.abs(numpy.sin((t*numpy.pi*2)**2))*9).astype(numpy.uint32)

            dst[0] = 9
            dst[2] = 5


            edges = sorted(list(zip(src,dst)), key=lambda t:t[1])
            edges = numpy.array(edges, dtype=numpy.uint32).T







            g = sam.Graph(edges,10)

            print(edges)
            # g.print()
            
            target = numpy.array([1,4,9], dtype=numpy.uint32)

            samples_per_layer, edge_indices_between_layers, load_per_layer = sam.sample_neighbors(g, 3, 2, target)

            for l in range(len(edge_indices_between_layers)-1,-1,-1):
                print("-------------------------------------------")
                print(samples_per_layer[l+1])
                print(edges[:,edge_indices_between_layers[l]])
                print(samples_per_layer[l])
                print("-------------------------------------------")
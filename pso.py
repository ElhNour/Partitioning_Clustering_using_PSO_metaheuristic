"""Particle Swarm Optimized Clustering
Optimizing centroid using K-Means style. In hybrid mode will use K-Means to seed first particle's centroid
"""
import numpy as np

from particle import Particle


class ParticleSwarmOptimizedClustering:
    def __init__(self,
                 n_cluster: int,
                 data: np.ndarray,
                 n_particles: int= 20,
                 w: float=1,
                 dump_w : float = 0.99,
                 c1: float = 2,
                 c2: float = 2,
                 hybrid: bool = True,
                 max_iter: int = 100,
                 print_debug: int = 10):
        self.n_cluster = n_cluster
        self.n_particles = n_particles
        self.data = data
        self.max_iter = max_iter
        self.particles = []
        self.hybrid = hybrid
        self.w = w
        self.dump_w = dump_w
        self.c1 = c1
        self.c2 = c2
        self.print_debug = print_debug
        self.gbest_score = np.inf
        self.gbest_centroids = None
        self.gbest_sse = np.inf
        self.gbest_quantization =np.inf
        self._init_particles()

    def _init_particles(self):
        for i in range(self.n_particles):
            particle = None
            if i == 0 and self.hybrid:
                particle = Particle(self.n_cluster, self.data,w=self.w, dump_w=self.dump_w,c1=self.c1,c2=self.c2, use_kmeans=True)
            else:
                particle = Particle(self.n_cluster, self.data,w=self.w,dump_w=self.dump_w,c1=self.c1,c2=self.c2, use_kmeans=False)
            if particle.best_score < self.gbest_score:
                self.gbest_centroids = particle.centroids.copy()
                self.gbest_score = particle.best_score
            self.particles.append(particle)
            self.gbest_sse = min(particle.best_sse, self.gbest_sse)
            self.gbest_quantization = min(particle.best_quantization,self.gbest_quantization)

    def run(self):
        print('Initial global best score', self.gbest_score)
        history = []
        for i in range(self.max_iter):
            for particle in self.particles:
                particle.update(self.gbest_centroids, self.data)
                # print(i, particle.best_score, self.gbest_score)
            for particle in self.particles:
                if particle.best_score < self.gbest_score:
                    self.gbest_centroids = particle.centroids.copy()
                    self.gbest_score = particle.best_score
                    self.gbest_quantization = particle.best_quantization
            history.append(self.gbest_score)
            if i % self.print_debug == 0:
                print('Iteration {:04d}/{:04d} current gbest score {:.18f}'.format(
                    i + 1, self.max_iter, self.gbest_score))
        print('Finish with gbest score {:.18f}'.format(self.gbest_score))
        return history


if __name__ == "__main__":
    pass
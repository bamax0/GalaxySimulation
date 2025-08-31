#include "LastTreeNodeUsingCuda.hpp"
#include <cuda_runtime.h>
#include <numeric>

namespace cudaTest {

__global__ static void computeForceCuda(Star s, Star* stars, size_t nbStars, Vec3* forces)
{
	size_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= nbStars) return;
	forces[id] = computeForce(s, stars[id]);

	if (id >= nbStars / 2) return; // On suppose nb stars = puissance de 2
	__syncthreads();
	forces[id] = forces[id] + forces[nbStars / 2 + id];

	// if (id > nbStars / 4) return; // On suppose nb stars = puissance de 4
	// forces[id] = forces[id] + forces[2 * id];

	return;
}

}
void LastTreeNodeUsingCuda::reset(const Bbox& bbox) {
	m_stars.clear();
	m_nbStars = 0u;
	if (m_gpuStarsPtr != nullptr)
	{
		cudaFree(m_gpuStarsPtr);
		m_gpuStarsPtr = nullptr;
	}
	if (m_forces != nullptr)
	{
		cudaFree(m_forces);
		m_forces = nullptr;
	}
};
Vec3 LastTreeNodeUsingCuda::computeTotalForce(const Star& star) const
{
	if (m_nbStars < 1000)
	{
		return computeTotalForceWithoutCuda(star);
	}

	if (m_gpuStarsPtr == nullptr) initCUda();
	// if (m_stars.size() > 2)
	// {
	// 	std::cout << m_stars.size() << std::endl;
	// }

	int threadsPerBlock = 128;
	int blocksPerGrid = (m_nbStars + threadsPerBlock - 1) / threadsPerBlock;
	cudaTest::computeForceCuda<<<blocksPerGrid, threadsPerBlock>>>(star, m_gpuStarsPtr, m_nbStars, m_forces);

	std::vector<Vec3> forces(m_nbStars / 2);
	cudaMemcpy(forces.data(), m_forces, m_nbStars/2 * sizeof(Vec3), cudaMemcpyDeviceToHost);

	Vec3 force = std::accumulate(forces.begin(), forces.end(), Vec3(0., 0., 0.));
	return force;
}
void LastTreeNodeUsingCuda::initCUda() const
{
	size_t sizeOfMem = m_nbStars * sizeof(Star);
	cudaMalloc((void**)&m_gpuStarsPtr, sizeOfMem);
	cudaMalloc((void**)&m_forces, m_nbStars *sizeof(Vec3));
	cudaMemcpy(m_gpuStarsPtr, m_stars.data(), sizeOfMem, cudaMemcpyHostToDevice);
}
#include "Algorithm/CudaSimpleAlgorithm.hpp"
#include <cuda_runtime.h>
#include <iostream>

__global__ void computeForces(Star* stars, size_t nbStars, Vec3* forces)
{
	size_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= nbStars) return;
	Vec3 f{ 0.f, 0.f, 0.f };
	for (size_t i{ 0u }; i < nbStars; ++i)
	{
		if (id != i)
		{
			f += computeForce(stars[id], stars[i]);
		}
	}
	Star blackHole;
	blackHole.position = { 0, 0, 0 };
	blackHole.mass = 2e6;
	f += computeForce(stars[id], blackHole);
	forces[id] = f;
	return;
}

__global__ void updatePositions(Star* stars, size_t nbStars, TimeS dt, Vec3* forces)
{
	size_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= nbStars) return;
	leapFrog(stars[id], getAcceleration(stars[id], forces[id]), dt);
}

GalaxySim::CudaSimpleAlgorithm::~CudaSimpleAlgorithm()
{
	cudaFree(m_gpuStarsPtr);
	cudaFree(m_gpuForcesPtr);
}

void GalaxySim::CudaSimpleAlgorithm::initWithGPU(size_t nbStars, Star* stars)
{
	m_nbStars = nbStars;
	m_gpuStarsPtr = stars;
	cudaMalloc((void**)&m_gpuForcesPtr, m_nbStars * sizeof(Vec3));
}

void GalaxySim::CudaSimpleAlgorithm::init(const std::vector<Star>& stars)
{
	if (m_gpuStarsPtr != nullptr)
	{
		cudaFree(m_gpuStarsPtr);
	}
	if (m_gpuForcesPtr != nullptr)
	{
		cudaFree(m_gpuForcesPtr);
	}
	m_nbStars = stars.size();
	size_t sizeOfMem = m_nbStars * sizeof(Star);
	cudaMalloc((void**)&m_gpuStarsPtr, sizeOfMem);
	cudaMalloc((void**)&m_gpuForcesPtr, m_nbStars * sizeof(Vec3));
	cudaMemcpy(m_gpuStarsPtr, stars.data(), sizeOfMem, cudaMemcpyHostToDevice);
}

void GalaxySim::CudaSimpleAlgorithm::next(TimeY dt)
{
	int threadsPerBlock = 128;
	int blocksPerGrid = (m_nbStars + threadsPerBlock-1) / threadsPerBlock;

	// std::cout << m_nbStars << std::endl;
	//std::cout << "Début calcule force" << std::endl;
	computeForces<<<blocksPerGrid, threadsPerBlock>>>(m_gpuStarsPtr, m_nbStars, m_gpuForcesPtr);

	//std::cout << "Fin calcule force" << std::endl;
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
	}
	cudaDeviceSynchronize();	
	cudaError_t err3 = cudaGetLastError();
	if (err3 != cudaSuccess) {
		printf("Kernel launch failed3: %s\n", cudaGetErrorString(err3));
	}
	updatePositions<<<blocksPerGrid, threadsPerBlock>>>(m_gpuStarsPtr, m_nbStars, dt, m_gpuForcesPtr);
}

std::vector<Star> GalaxySim::CudaSimpleAlgorithm::getStars() const
{
	std::vector<Star> stars(m_nbStars, Star{});
	cudaMemcpy(stars.data(), m_gpuStarsPtr, m_nbStars * sizeof(Star), cudaMemcpyDeviceToHost);
	cudaError_t err3 = cudaGetLastError();
	if (err3 != cudaSuccess) {
		printf("Erreure a la récupération des étoile: %s\n", cudaGetErrorString(err3));
	}
	return stars;
}

const Star* GalaxySim::CudaSimpleAlgorithm::getCudaStars() const
{
	return m_gpuStarsPtr;
}

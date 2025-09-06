#include "CudaBarnesHutAlgorithm.hpp"
#include "Physics/function.hpp"
#include <execution>
namespace CudaBarnesHut
{
	__global__ static void computeForces(Star* stars, size_t nbStars, Vec3* forces, OctreePtr octreeNodes)
	{
		size_t id = threadIdx.x + blockIdx.x * blockDim.x;
		if (id >= nbStars) return;
		Vec3 f{ octreeNodes.nodes[0].computeTotalForceWithoutRec(octreeNodes, stars[id]) };
		forces[id] = f;
		return;
	}

	__global__ static void updatePositions(Star* stars, size_t nbStars, TimeS dt, Vec3* forces)
	{
		size_t id = threadIdx.x + blockIdx.x * blockDim.x;
		if (id >= nbStars) return;
		leapFrog(stars[id], getAcceleration(stars[id], forces[id]), dt);
	}
}

namespace GalaxySim
{
	CudaBarnesHutAlgorithm::~CudaBarnesHutAlgorithm()
	{
		delete m_firstNode;
		cudaFree(m_gpuStarsPtr);
		cudaFree(m_gpuForcesPtr);
	}

	void CudaBarnesHutAlgorithm::init(const std::vector<Star>& stars)
	{
		// m_stars = stars;
		// m_rootNode->reset(computeBbox());
		if(m_gpuStarsPtr != nullptr) cudaFree(m_gpuStarsPtr);
		if(m_gpuForcesPtr != nullptr) cudaFree(m_gpuForcesPtr);

		m_nbStars = stars.size();
		size_t sizeOfMem = m_nbStars * sizeof(Star);
		cudaMalloc((void**)&m_gpuStarsPtr, sizeOfMem);
		cudaMalloc((void**)&m_gpuForcesPtr, m_nbStars * sizeof(Vec3));
		cudaMemcpy(m_gpuStarsPtr, stars.data(), sizeOfMem, cudaMemcpyHostToDevice);
	}

	void CudaBarnesHutAlgorithm::next(TimeY dt)
	{
		//auto start = std::chrono::steady_clock::now();
		m_firstNode->reset(computeBbox());

		std::vector<Star> stars = getStars();
		m_firstNode->startInserting();
		for (const Star& star : stars)
		{
			m_firstNode->appendStar(star);
		}
		m_firstNode->endInserting();
		// 
		// stars.clear(); // Libération de la mémoire

		// std::set<size_t> idxVec;
		// std::cout << "Check des indices : " << m_rootNode->fillIndex(idxVec) << std::endl;

		// auto end = std::chrono::steady_clock::now();
		// 
		// // Store the time difference between start and end
		// auto diff = end - start;
		// std::cout << "Init de l'arbe: " << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;
		// start = std::chrono::steady_clock::now();


		OctreePtr flattenOctree = m_rootNode->createCudaPtr();

		cudaError_t err1 = cudaGetLastError();
		if (err1 != cudaSuccess) {
			printf("Kernel launch failed1: %s\n", cudaGetErrorString(err1));
		}
		// m_firstNode->reset({}); //	Libération de la mémoire

		int threadsPerBlock = 128;
		int blocksPerGrid = (m_nbStars + threadsPerBlock - 1) / threadsPerBlock;
		CudaBarnesHut::computeForces<<<blocksPerGrid, threadsPerBlock>>>(m_gpuStarsPtr, m_nbStars, m_gpuForcesPtr, flattenOctree);
		cudaDeviceSynchronize();
		// std::cout << "fin calcul force" << std::endl;

		cudaFree(flattenOctree.nodes);
		cudaFree(flattenOctree.lastNodes);
		flattenOctree.nodes = nullptr;
		flattenOctree.lastNodes = nullptr;

		cudaError_t err2 = cudaGetLastError();
		if (err2 != cudaSuccess) {
			printf("Kernel launch failed2: %s\n", cudaGetErrorString(err2));
		}
		// std::cout << "Début mise a jour possition" << std::endl;
		CudaBarnesHut::updatePositions<<<blocksPerGrid, threadsPerBlock>>>(m_gpuStarsPtr, m_nbStars, dt, m_gpuForcesPtr);
		cudaDeviceSynchronize();

		cudaError_t err3 = cudaGetLastError();
		if (err3 != cudaSuccess) {
			printf("Kernel launch failed3: %s\n", cudaGetErrorString(err3));
		}

		// end = std::chrono::steady_clock::now();
		// diff = end - start;
		// std::cout << "Calcul de la force: " << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;
	}


	std::vector<Star> CudaBarnesHutAlgorithm::getStars() const
	{
		std::vector<Star> stars(m_nbStars, Star{});
		cudaMemcpy(stars.data(), m_gpuStarsPtr, m_nbStars * sizeof(Star), cudaMemcpyDeviceToHost);

		cudaError_t err3 = cudaGetLastError();
		if (err3 != cudaSuccess) {
			printf("Kernel launch  getStars: %s\n", cudaGetErrorString(err3));
		}
		return stars;
	}

	Bbox CudaBarnesHutAlgorithm::computeBbox() const
	{
		DistanceLy xMin{ std::numeric_limits<float>::max() };
		DistanceLy yMin{ std::numeric_limits<float>::max() };
		DistanceLy zMin{ std::numeric_limits<float>::max() };

		DistanceLy xMax{-std::numeric_limits<float>::max() };
		DistanceLy yMax{-std::numeric_limits<float>::max() };
		DistanceLy zMax{-std::numeric_limits<float>::max() };

		for (const Star& star : getStars())
		{
			xMin = std::min(star.position.x, xMin);
			yMin = std::min(star.position.y, yMin);
			zMin = std::min(star.position.z, zMin);

			xMax = std::max(star.position.x, xMax);
			yMax = std::max(star.position.y, yMax);
			zMax = std::max(star.position.z, zMax);
		}

		DistanceLy size{ std::max(std::max(xMax - xMin, yMax - yMin), zMax - zMin) / 2.f +0.1f};
		return Bbox(PosLy{
				(xMin + xMax) / 2.f,
				(yMin + yMax) / 2.f,
				(zMin + zMax) / 2.f
			}, size);
	}

	void GalaxySim::CudaBarnesHutAlgorithm::initWithGPU(size_t nbStars, Star* stars)
	{
		m_nbStars = nbStars;
		m_gpuStarsPtr = stars;
		cudaMalloc((void**)&m_gpuForcesPtr, m_nbStars * sizeof(Vec3));
	}
}
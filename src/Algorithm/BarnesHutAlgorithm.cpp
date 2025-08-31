#include "BarnesHutAlgorithm.hpp"
#include "Physics/function.hpp"
#include <execution>

namespace GalaxySim
{
	void BarnesHutAlgorithm::init(const std::vector<Star>& stars)
	{
		m_stars = stars;
		// m_rootNode->reset(computeBbox());
	}

	void BarnesHutAlgorithm::next(TimeY dt)
	{
		// auto start = std::chrono::steady_clock::now();
		m_rootNode->reset(computeBbox());

		m_rootNode->startInserting();
		for (const Star& star : m_stars)
		{
			m_rootNode->appendStar(star);
		}
		m_rootNode->endInserting();


		// auto end = std::chrono::steady_clock::now();
		// Store the time difference between start and end
		// auto diff = end - start;
		// double timeInit = std::chrono::duration<double, std::milli>(diff).count();
		// 
		// start = std::chrono::steady_clock::now();

		std::for_each(
			std::execution::par,
			m_stars.begin(),
			m_stars.end(),
			[this, dt](Star& star)
			{
				Vec3 acc = getAcceleration(star, m_rootNode->computeTotalForce(star));
				leapFrog(star, acc, dt);
			});

		// end = std::chrono::steady_clock::now();
		// diff = end - start;
		// std::cout << "Init octree: " << timeInit << "ms | Calcul de la force: " << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;
	}

	Bbox BarnesHutAlgorithm::computeBbox() const
	{
		float xMin{ std::numeric_limits<float>::max() };
		float yMin{ std::numeric_limits<float>::max() };
		float zMin{ std::numeric_limits<float>::max() };

		float xMax{-std::numeric_limits<float>::max() };
		float yMax{-std::numeric_limits<float>::max() };
		float zMax{-std::numeric_limits<float>::max() };

		for (const Star& star : m_stars)
		{
			xMin = std::min(star.position.x, xMin);
			yMin = std::min(star.position.y, yMin);
			zMin = std::min(star.position.z, zMin);

			xMax = std::max(star.position.x, xMax);
			yMax = std::max(star.position.y, yMax);
			zMax = std::max(star.position.z, zMax);
		}


		float size{ std::max(std::max(xMax - xMin, yMax - yMin), zMax - zMin) / 2.f +0.1f};
		return Bbox({
				(xMin + xMax) / 2.f,
				(yMin + yMax) / 2.f,
				(zMin + zMax) / 2.f
			}, size);
	}
}
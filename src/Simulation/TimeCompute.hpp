#pragma once
#include <GalaxyGenerator/GalaxyGenerator.hpp>
#include <Algorithm/AlgorithmInterface.hpp>
#include <Physics/constant.hpp>
#include <Physics/Star.hpp>
#include <Render/ImageGen.hpp>
#include <Utils/Chrono.hpp>


class Simulation
{
private:
	size_t m_nbStar{ 1000 };
	std::vector<Star> m_stars;
	GenParam m_param{};
	ImageGen m_gen;

	TimeY m_yearEnd{ static_cast<float>(UNIVERS_AGE / YEAR / 5.) };
	size_t m_nbImage{ 5000 };
public:

	Simulation() : m_gen()
	{
		DistanceLy sizeMilky = static_cast<float>(MILKYWAY_SIZE / LIGHT_YEAR);
		// param.min = -sizeMilky *0.5;
		m_param.max = sizeMilky * 0.8;
		// param.maxMass = 100000. * 10;
		// param.minMass = 0.1 * 100;
		m_param.maxSpeed = 0.8;
		// param.minSpeed = 0;

		auto projectDir = std::filesystem::absolute(std::filesystem::path(u8"..") / u8".." / u8".." / u8"..");
		m_gen.init(projectDir / u8"test_data" / u8"output");
	}
	const GenParam& getParam() const { return m_param; }
	void setNbStars(size_t nbStar)
	{
		m_nbStar = nbStar;
	}
	void setNbStep(size_t nbStep)
	{
		m_nbImage = nbStep; // Pour le moment, on genere autant d'image que de generation
	}
	void setYearEnd(TimeY end)
	{
		m_yearEnd = end;
	}
	void generateRandomStars()
	{
		double M = MILKYWAY_MASS / SOLAR_MASS;          // total mass of the galaxy
		double starMass = M / m_nbStar;
		m_param.minMass = starMass;
		m_param.maxMass = starMass;
		m_stars = getSquareUniversFromGPUToCPU(m_nbStar, m_param);

	}
	void generateGalaxyStars()
	{
		m_stars = getPlummerGalaxyFromGPUToCpu(m_nbStar, m_param);
	}

	void simulateGalaxy(GalaxySim::AlgorithmInterface* algo)
	{
		m_gen.reinit();
		std::vector<Star> copyStars = m_stars;
		algo->init(copyStars);
		TimeY dt = m_yearEnd / (m_nbImage * 1.f);
		TimeY dtGenImg = m_yearEnd / m_nbImage;
		TimeY nexSaveImg = 0.f;
		size_t cpt{ 0u };
		std::cout << "Initialisation" << std::endl;
		TestChrono::start();
		for (TimeY time{ 0.f }; time < m_yearEnd; time += dt)
		{
			if (nexSaveImg <= time)
			{
				TestChrono::end();
				std::cout << "|| Generation de l'image n° " << cpt+1 << ", temps depuis la dernier image: " << TestChrono::getTime() << std::endl;
				m_gen.generateImg(algo->getStars(), {}, m_param.max * 3, cpt);
				nexSaveImg += dtGenImg;
				++cpt;
				TestChrono::start();
			}
			algo->next(dt);
		}
		std::cout << "Fin" << std::endl;
		TestChrono::end();
		TestChrono::show();
	}

	double getSimulationTime(GalaxySim::AlgorithmInterface* algo)
	{
		std::vector<Star> copyStars = m_stars;
		algo->init(copyStars);
		TimeY dt = m_yearEnd / (m_nbImage);

		TestChrono::start();
		for (TimeY time{ 0.f }; time < m_yearEnd; time += dt)
		{
			algo->next(dt);
		}
		TestChrono::end();
		return TestChrono::getTime();
	}
};
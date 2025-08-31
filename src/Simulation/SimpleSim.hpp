#pragma once
#include <Algorithm/CudaSimpleAlgorithm.hpp>
#include <GalaxyGenerator/GalaxyGenerator.hpp>
#include <Physics/constant.hpp>
#include <Physics/Star.hpp>
#include <Render/ImageGen.hpp>
#include <Utils/Chrono.hpp>
/*
void fGPU()
{
	size_t nbStars{ static_cast<size_t>(1e3) };
	GenParam param{};
	DistanceLy sizeMilky = static_cast<float>(MILKYWAY_SIZE / LIGHT_YEAR);
	// param.min = -sizeMilky *0.5;
	param.max = sizeMilky * 0.8;
	// param.maxMass = 100000. * 10;
	// param.minMass = 0.1 * 100;
	param.maxSpeed = 0.8;
	// param.minSpeed = 0;

	ImageGen gen;
	auto projectDir = std::filesystem::absolute(std::filesystem::path(u8"..") / u8".." / u8".." / u8"..");
	gen.init(projectDir / u8"test_data" / u8"output");

	Star* starsGPU = getPlummerGalaxyFromGPU(nbStars, param);

	GalaxySim::CudaSimpleAlgorithm alg{};
	alg.initWithGPU(nbStars, starsGPU);

	TimeY yearEnd =static_cast<float>(UNIVERS_AGE / YEAR / 5.);

	size_t nbImage{ 5000 };
	TimeY dt = yearEnd / (nbImage * 1.f);

	TimeY dtGenImg = yearEnd / nbImage;
	TimeY nexSaveImg = 0.f;
	size_t cpt{ 0u };
	std::cout << "Init" << std::endl;
	TestChrono::start();

	// std::cout << "\033[32m";

	for (TimeY time{ 0.f }; time< yearEnd; time += dt)
	{
		// std::cout << "Gen " << time << std::endl;
		if (nexSaveImg <= time)
		{
			std::cout << "||| step de la simu physique " << cpt << std::endl;
			TestChrono::end();
			TestChrono::show();

			gen.generateImg(alg.getStars(), {}, param.max*3, cpt);
			nexSaveImg += dtGenImg;
			++cpt;

			TestChrono::start();
		}
		alg.next(dt);
	}
	std::cout << "Fin" << std::endl;
	TestChrono::end();
	TestChrono::show();

}
void fCPU()
{
	size_t nbStars{ static_cast<size_t>(1e9) };
	GenParam param{};
	param.min = -MILKYWAY_SIZE;
	param.max = MILKYWAY_SIZE;
	param.maxMass = 100000. * SOLAR_MASS;
	param.minMass = 0.1 * SOLAR_MASS * 100;
	param.maxSpeed = 0.1 * LIGTH_SPEED;
	param.minSpeed = 0;

	ImageGen gen;
	auto projectDir = std::filesystem::absolute(std::filesystem::path(u8"..") / u8".." / u8".." / u8"..");
	gen.init(projectDir / u8"test_data" / u8"output");

	std::vector<Star> starsCPU = getSquareUniversFromGPUToCPU(nbStars, param);

	GalaxySim::CudaSimpleAlgorithm alg{};
	// alg.initWithGPU(nbStars, starsGPU);

	TimeS yearEnd = UNIVERS_AGE;

	size_t nbImage{ 5000 };
	TimeS dt = yearEnd / (nbImage * 200);

	TimeS dtGenImg = yearEnd / nbImage;
	TimeS nexSaveImg = 0;
	size_t cpt{ 0u };
	std::cout << "Init" << std::endl;
	TestChrono::start();
	for (TimeS time{ 0 }; time< yearEnd; time += dt)
	{
		// std::cout << "Gen " << time << std::endl;
		if (nexSaveImg <= time)
		{
			std::cout << "Debut sauvegarde image " << cpt << std::endl;
			TestChrono::end();
			TestChrono::show();

			TestChrono::start();
			gen.generateImg(starsCPU, {}, param.max*1.2, cpt);
			nexSaveImg += dtGenImg;
			++cpt;

			std::cout << "Fin sauvegarde" << std::endl;
			TestChrono::end();
			TestChrono::show();
			TestChrono::start();
		}
		alg.next(dt);
	}
	std::cout << "Fin" << std::endl;
	TestChrono::end();
	TestChrono::show();

}
*/
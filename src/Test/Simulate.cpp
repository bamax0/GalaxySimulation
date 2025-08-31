#include <gtest/gtest.h>
#include <Physics/Star.hpp>
#include <Algorithm/CudaSimpleAlgorithm.hpp>
#include <GalaxyGenerator/GalaxyGenerator.hpp>
#include <Physics/constant.hpp>
#include <Render/ImageGen.hpp>

TEST(Simulation, squaredUniverseGPU)
{
	size_t nbStars{ static_cast<size_t>(1e8) };
	GenParam param{};
	param.min = -1.;
	param.max = 1.;
	param.maxMass = 100000.;
	param.minMass = 0.1;
	param.maxSpeed = 0.1;
	param.minSpeed = 0;

	ImageGen gen;
	auto projectDir = std::filesystem::absolute(std::filesystem::path(u8"..") / u8".." / u8".." / u8"..");
	gen.init(projectDir / u8"test_data" / u8"output");

	Star* starsGPU = getSquareUniversFromGPU(nbStars, param);
	GalaxySim::CudaSimpleAlgorithm alg{};
	alg.initWithGPU(nbStars, starsGPU);

	TimeS yearEnd = static_cast<TimeS>(UNIVERS_AGE / YEAR);
	TimeS dt = 100000;

	size_t nbImage{ 1000 };
	TimeS dtGenImg = yearEnd / nbImage;
	TimeS nexSaveImg = 0;
	size_t cpt{ 0u };
	std::cout << "Init" << std::endl;
	for (TimeS time{ 0 }; time< yearEnd; ++time)
	{
		std::cout << "Gen " << time << std::endl;
		alg.next(dt);
		if (nexSaveImg <= time)
		{
			std::cout << "SaveImg " << cpt << std::endl;
			gen.generateImg(alg.getStars(), {}, 0, cpt);
			nexSaveImg += dtGenImg;
			++cpt;
		}
	}

}
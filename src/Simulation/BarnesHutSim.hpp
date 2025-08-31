#pragma once
#include <GalaxyGenerator/GalaxyGenerator.hpp>
#include <Algorithm/CudaBarnesHutAlgorithm.hpp>
#include <Algorithm/BarnesHutAlgorithm.hpp>
#include <Algorithm/Nodes/OctreeNode.hpp>
#include <Algorithm/Nodes/FirstTreeNode.hpp>
#include <Algorithm/Nodes/CudaTreeNode.hpp>
#include <Physics/constant.hpp>
#include <Physics/Star.hpp>
#include <Render/ImageGen.hpp>
#include <Utils/Chrono.hpp>

void barnesHut()
{
	size_t nbStars{ static_cast<size_t>(1e5) };
	GenParam param{};
	DistanceLy sizeMilky = static_cast<float>(MILKYWAY_SIZE / LIGHT_YEAR);
	// param.min = -sizeMilky *0.5;
	param.max = sizeMilky * 0.8;
	// param.maxMass = 100000. * 10;
	// param.minMass = 0.1 * 100;
	param.maxSpeed = 0.1;
	// param.minSpeed = 0;

	ImageGen gen;
	auto projectDir = std::filesystem::absolute(std::filesystem::path(u8"..") / u8".." / u8".." / u8"..");
	gen.init(projectDir / u8"test_data" / u8"output");

	std::vector<Star> stars = getPlummerGalaxyFromGPUToCpu(nbStars, param);

	OctreeNode* node = new OctreeNode(Bbox{ {}, param.max * 2.01f });
	FirstTreeNode<8>* firstNode = new FirstTreeNode<8>(static_cast<SimplifiedTreeNode<8>*>(node));
	// SedecTreeNode* node = new SedecTreeNode(Bbox{ {}, param.max  * 2.01f}); Tres ineficace pas utiliser
	// FirstTreeNode<64>* firstNode = new FirstTreeNode<64>(static_cast<TreeNode<64>*>(node));
	GalaxySim::BarnesHutAlgorithm alg{ firstNode };
	alg.init(stars);

	TimeY yearEnd = static_cast<float>(UNIVERS_AGE / YEAR / 10.);

	size_t nbImage{ 2000 };
	TimeY dt = yearEnd / (nbImage * 5.f);

	TimeY dtGenImg = yearEnd / nbImage;
	TimeY nexSaveImg = 0.f;
	size_t cpt{ 0u };
	std::cout << "Init" << std::endl;
	TestChrono::start();

	// std::cout << "\033[32m";

	for (TimeY time{ 0.f }; time < yearEnd; time += dt)
	{
		// std::cout << "Gen " << time << std::endl;
		if (nexSaveImg <= time)
		{
			std::cout << "||| step de la simu physique " << cpt << std::endl;
			TestChrono::end();
			TestChrono::show();
			// std::cout << "Debut sauvegarde image " << cpt << std::endl;

			// TestChrono::start();
			gen.generateImg(alg.getStars(), {}, param.max * 3, cpt);
			nexSaveImg += dtGenImg;
			++cpt;
			// std::cout << "\033[31m";
			// std::cout << "Fin sauvegarde" << std::endl;
			// TestChrono::end();
			// TestChrono::show();
			TestChrono::start();
			// std::cout << "\033[32m";
		}
		alg.next(dt);
	}
	std::cout << "Fin" << std::endl;
	TestChrono::end();
	TestChrono::show();
}

void barnesHutCuda()
{
	size_t nbStars{ static_cast<size_t>(1e5) };
	GenParam param{};
	DistanceLy sizeMilky = static_cast<float>(MILKYWAY_SIZE / LIGHT_YEAR);
	// param.min = -sizeMilky *0.5;
	param.max = sizeMilky * 0.8;
	// param.maxMass = 100000. * 10;
	// param.minMass = 0.1 * 100;
	param.maxSpeed = 0.1;
	param.minSpeed = 0;

	ImageGen gen;
	auto projectDir = std::filesystem::absolute(std::filesystem::path(u8"..") / u8".." / u8".." / u8"..");
	gen.init(projectDir / u8"test_data" / u8"output");

	Star* stars = getPlummerGalaxyFromGPU(nbStars, param);

	GalaxySim::CudaBarnesHutAlgorithm alg{ Bbox{ {0., 0., 0.}, param.max * 2.01f} };
	alg.initWithGPU(nbStars, stars);

	TimeY yearEnd = static_cast<float>(UNIVERS_AGE / YEAR / 10.);

	size_t nbImage{ 2000 };
	TimeY dt = yearEnd / (nbImage * 5.f);

	TimeY dtGenImg = yearEnd / nbImage;
	TimeY nexSaveImg = 0.f;
	size_t cpt{ 0u };
	std::cout << "Init" << std::endl;
	TestChrono::start();

	for (TimeY time{ 0.f }; time < yearEnd; time += dt)
	{
		if (nexSaveImg <= time)
		{
			std::cout << "||| step de la simu physique " << cpt << std::endl;
			TestChrono::end();
			TestChrono::show();
			gen.generateImg(alg.getStars(), {}, param.max * 3, cpt);
			nexSaveImg += dtGenImg;
			++cpt;
			std::cout << "Fin sauvegarde" << std::endl;
			TestChrono::start();
		}
		alg.next(dt);
	}
	std::cout << "Fin" << std::endl;
	TestChrono::end();
	TestChrono::show();
}

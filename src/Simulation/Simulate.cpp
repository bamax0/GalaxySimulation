#include "BarnesHutSim.hpp"
#include "SimpleSim.hpp"
#include "TimeCompute.hpp"
#include <Algorithm/Nodes/SedecTreeNode.hpp>
#include <map>
#include <fstream>

void saveCsv(const std::string& filename,
	const std::map<size_t, std::map<std::string, double>>& timeMapRdm)
{
	// Récupération de tous les noms de colonnes (union des clés internes)
	std::set<std::string> allColumns;
	for (const auto& [rowIndex, rowMap] : timeMapRdm) {
		for (const auto& [colName, value] : rowMap) {
			allColumns.insert(colName);
		}
	}

	// Ouverture fichier
	std::ofstream out{ filename };
	if (!out.is_open()) {
		throw std::runtime_error("Impossible d'ouvrir le fichier CSV");
	}

	// Écriture de l'entête
	out << "Index";
	for (const auto& colName : allColumns) {
		out << ";" << colName;
	}
	out << "\n";

	// Écriture des lignes
	for (const auto& [rowIndex, rowMap] : timeMapRdm) {
		out << rowIndex;
		for (const auto& colName : allColumns) {
			auto it = rowMap.find(colName);
			if (it != rowMap.end()) {
				out << ";" << it->second;
			}
			else {
				out << ";"; // vide si pas de valeur pour cette colonne
			}
		}
		out << "\n";
	}
}

void computeAllTime()
{
	Simulation sim{};

	Bbox bbox{ {0., 0., 0.}, sim.getParam().max * 2.01f };

	// BarnesHut
	// 64
	SedecTreeNode* node64 = new SedecTreeNode(bbox);
	FirstTreeNode<64>* firstNode64 = new FirstTreeNode<64>(static_cast<TreeNode<64>*>(node64));
	GalaxySim::BarnesHutAlgorithm barnesHutCPU64{ firstNode64 };
	// 8
	OctreeNode* node = new OctreeNode(bbox);
	FirstTreeNode<8>* firstNode = new FirstTreeNode<8>(static_cast<SimplifiedTreeNode<8>*>(node));
	GalaxySim::BarnesHutAlgorithm barnesHutCPU8{ firstNode };
	GalaxySim::CudaBarnesHutAlgorithm barnesHutCuda{ bbox };

	// Simple
	GalaxySim::CudaSimpleAlgorithm cudaSimple{};

	sim.setYearEnd(static_cast<float>(UNIVERS_AGE / YEAR / 1000.));
	sim.setNbStep(10);

	std::map<size_t, std::map<std::string, double>> timeMapRdm{};

	size_t nbStarMin = 10000;
	size_t nbStarMax = 5000000;
	size_t nbStarStep = 249500; // 20 interval
	for (size_t i{ nbStarMin }; i < nbStarMax; i += nbStarStep)
	{
		std::cout << "Début de la génération pour " << i << " étoiles" << std::endl;
		sim.setNbStars(i);
		sim.generateRandomStars();

		double barnesHut8CPUTime = sim.getSimulationTime(&barnesHutCPU8);
		timeMapRdm[i]["BC8"] = barnesHut8CPUTime;
		double barnesHut8GPUTime = sim.getSimulationTime(&barnesHutCuda);
		timeMapRdm[i]["BG8"] = barnesHut8GPUTime;
		double barnesHut64CPUTime = 0;
		//double barnesHut64CPUTime = sim.getSimulationTime(&barnesHutCPU64);
		//timeMapRdm[i]["BC64"] = barnesHut8CPUTime;
		double simpleGPUTime = sim.getSimulationTime(&cudaSimple);
		timeMapRdm[i]["S"] = simpleGPUTime;
		std::cout << "Simple: " << simpleGPUTime << "ms | Barnes-hut 64 :" << barnesHut64CPUTime << "ms | cpu 8:" << barnesHut8CPUTime << "ms | gpu 8: " << barnesHut8GPUTime << "ms" << std::endl;
	}
	saveCsv("../../../../rdmTime.csv", timeMapRdm);
	timeMapRdm.clear();
	for (size_t i{ nbStarMin }; i < nbStarMax; i += nbStarStep)
	{
		std::cout << "Début de la génération pour " << i << " étoiles" << std::endl;
		sim.setNbStars(i);
		sim.generateGalaxyStars();

		double barnesHut8CPUTime = sim.getSimulationTime(&barnesHutCPU8);
		timeMapRdm[i]["BC8"] = barnesHut8CPUTime;
		double barnesHut8GPUTime = sim.getSimulationTime(&barnesHutCuda);
		timeMapRdm[i]["BG8"] = barnesHut8GPUTime;
		double barnesHut64CPUTime = 0;
		//double barnesHut64CPUTime = sim.getSimulationTime(&barnesHutCPU64);
		//timeMapRdm[i]["BC64"] = barnesHut8CPUTime;
		double simpleGPUTime = sim.getSimulationTime(&cudaSimple);
		timeMapRdm[i]["S"] = simpleGPUTime;
		std::cout << "Simple: " << simpleGPUTime << "ms | Barnes-hut 64 :" << barnesHut64CPUTime << "ms | cpu 8:" << barnesHut8CPUTime << "ms | gpu 8: " << barnesHut8GPUTime << "ms" << std::endl;
	}
	saveCsv("../../../../galaxyTime.csv", timeMapRdm);
}


int main()
{
	//barnesHutCuda();
	 barnesHut();
	// fGPU();
	// fCPU();

	//computeAllTime();

	return 0;
}
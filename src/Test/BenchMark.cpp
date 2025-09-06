#include <gtest/gtest.h>
#include <Physics/Star.hpp>
#include <Algorithm/CudaSimpleAlgorithm.hpp>
#include <Algorithm/CpuSimpleAlgorithm.hpp>
#include <Algorithm/BarnesHutAlgorithm.hpp>
#include <Algorithm/Nodes/SedecTreeNode.hpp>
#include <Algorithm/Nodes/OctreeNode.hpp>
#include <GalaxyGenerator/GalaxyGenerator.hpp>
#include <chrono>

using hcc = std::chrono::high_resolution_clock;

class SimuBench : public testing::Test {
public:
    void simulate(GalaxySim::AlgorithmInterface* alg)
    {
        alg->init(m_stars);
        for (size_t i{ 0u }; i < m_count; ++i)
        {
            alg->next(1.);
        }
    }

protected:
    SimuBench() {
    }

    void SetUp() override {
        GenParam param{};
        param.min = -10000.;
        param.max = 10000.;
        param.maxMass = 100000.;
        param.minMass = 0.1;
        param.maxSpeed = 0.1;
        param.minSpeed = 0;
        m_stars = getSquareUniversFromGPUToCPU(m_nbStars, param);
        m_start = hcc::now();
    }

    void TearDown() override {
        // Code here will be called immediately after each test (right
        // before the destructor).
        m_start = hcc::now();		
        std::chrono::duration<double, std::milli> duration_ms = m_end - m_start;
        std::cout << "Fin du test: " << duration_ms.count() << " ms" << std::endl;
        std::cout << "Temps moyen de chanque appelle de next: " << duration_ms.count() / static_cast<double>(m_count) << " ms" << std::endl;

        // TODO enregisterer dans un fichier
    }

    std::vector<Star> m_stars;
    hcc::time_point m_start{};
    hcc::time_point m_end{};
    size_t m_count{ 10 };
    size_t m_nbStars{ static_cast<size_t>(1e5) };
};

TEST_F(SimuBench, SimpleCPU)
{
    std::cout << "Algo simple sur CPU" << std::endl;
    GalaxySim::CpuSimpleAlgorithm alg{};
    simulate(&alg);
}
TEST_F(SimuBench, SimpleGPU)
{
    std::cout << "Algo simple sur GPU" << std::endl;
    GalaxySim::CudaSimpleAlgorithm alg{};
    simulate(&alg);
}
TEST_F(SimuBench, BarnesHutOctreeCPU)
{
    std::cout << "Barnes-hut (octree) sur CPU" << std::endl;
    OctreeNode oct{ Bbox{} };
    GalaxySim::BarnesHutAlgorithm alg{&oct};
    simulate(&alg);
}

TEST_F(SimuBench, BarnesHutSedecTreeCPU)
{
    std::cout << "Barnes-hut (sedecTree) sur CPU" << std::endl;
    SedecTreeNode* node = new SedecTreeNode(Bbox{});
    GalaxySim::BarnesHutAlgorithm alg{ node };
    simulate(&alg);
}
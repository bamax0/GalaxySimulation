#include "GalaxyGenerator.hpp"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include "Physics/constant.hpp"

__device__ float randomNumber(curandState* state, float min, float max)
{
    return curand_uniform(state) * (max - min) + min;
}

__global__ void generateStarsGPU(const size_t nbStars, const GenParam param,
    Star* stars, const unsigned long long seed)
{
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= nbStars) return;

    curandState state;
    curand_init(seed, static_cast<unsigned long long>(id), 0, &state);

    stars[id] = Star{
        PosLy(
            randomNumber(&state, param.min, param.max),
            randomNumber(&state, param.min, param.max),
            randomNumber(&state, param.min, param.max)
        ),
        SpeedLs(
            randomNumber(&state, param.minSpeed, param.maxSpeed),
            randomNumber(&state, param.minSpeed, param.maxSpeed),
            randomNumber(&state, param.minSpeed, param.maxSpeed)
        ),
        AccelLyY2(0., 0., 0.),
        randomNumber(&state, param.minMass, param.maxMass)
        };
}

__global__ void generateStarsPlummerGPU(const size_t nbStars, const GenParam param,
    Star* stars, const unsigned long long seed, float M, float starMass)
{
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= nbStars) return;

    curandState state;
    curand_init(seed, static_cast<unsigned long long>(id), 0, &state);

    // https://stackoverflow.com/questions/28434765/how-to-randomly-distribute-n-masses-such-that-they-follow-a-plummer-density-dis
    float a = 1.f / 12.f;
    float mM = randomNumber(&state, 0.2, 0.99); // Proportion de mass entre l'etoile et le centre de la galaxy
    float radius = param.max * a / std::sqrt(std::pow(mM, -2.0f / 3.0f) - 1.f);

    float theta = randomNumber(&state, 0.f, 2.00001f * fPI);

    float massInside = mM * M;
    float rSpeed = param.maxSpeed * std::sqrt(REDUCED_Gf * massInside / radius);

    Star s;
    s.position = param.center + PosLy{ radius * std::cos(theta), radius * std::sin(theta), 0.f };
    s.speed = { -rSpeed * std::sin(theta), rSpeed * std::cos(theta), 0.f };
    s.acceleration = { 0.f, 0.f, 0.f };
    s.mass = randomNumber(&state, starMass * 0.0001f, starMass * 1.9999f);
    stars[id] = s;
}


Star* getSquareUniversFromGPU(const size_t& nbStars, const GenParam& param)
{
    Star* starsGpu;
    cudaMalloc((void**)&starsGpu, nbStars * sizeof(Star));

    int threadsPerBlock = 256;
    int blocksPerGrid = (nbStars + threadsPerBlock - 1) / threadsPerBlock;
    unsigned long long seed = static_cast<unsigned long long>(time(NULL));
    generateStarsGPU<<<blocksPerGrid, threadsPerBlock>>>(nbStars, param, starsGpu, seed);

    return starsGpu;
}

Star* getPlummerGalaxyFromGPU(const size_t& nbStars, GenParam param)
{
    Star* starsGpu;
    cudaMalloc((void**)&starsGpu, nbStars * sizeof(Star));

    int threadsPerBlock = 256;
    int blocksPerGrid = (nbStars + threadsPerBlock - 1) / threadsPerBlock;
    unsigned long long seed = static_cast<unsigned long long>(time(NULL));

    size_t N = nbStars;         // number of stars
    double M = MILKYWAY_MASS / SOLAR_MASS;          // total mass of the galaxy
    double starMass = M / N;

    generateStarsPlummerGPU<<<blocksPerGrid, threadsPerBlock>>>(nbStars, param, starsGpu, seed, M, starMass);

    return starsGpu;
}
     
std::vector<Star> getSquareUniversFromGPUToCPU(const size_t& nbStars, const GenParam& param)
{
    std::vector<Star> stars;
    stars.resize(nbStars);
    Star* starsGpu = getSquareUniversFromGPU(nbStars, param);
    cudaMemcpy(stars.data(), starsGpu, nbStars * sizeof(Star), cudaMemcpyDeviceToHost);
    cudaFree(starsGpu);
    return stars;
}

std::vector<Star> getPlummerGalaxyFromGPUToCpu(const size_t& nbStars, const GenParam& param)
{
    std::vector<Star> stars;
    stars.resize(nbStars);
    Star* starsGpu = getPlummerGalaxyFromGPU(nbStars, param);
    cudaMemcpy(stars.data(), starsGpu, nbStars * sizeof(Star), cudaMemcpyDeviceToHost);
    cudaFree(starsGpu);
    return stars;
}

 // std::vector<Star> getSquareUnivers(const size_t& nbStars, const GenParam& param)
 // {
 //     std::vector<Star> stars;
 //     for (size_t i{ 0u }; i < nbStars; ++i)
 //     {
 //         stars.push_back(Star{
 //             PosM(
 //                 glm::linearRand(param.min, param.max),
 //                 glm::linearRand(param.min, param.max),
 //                 glm::linearRand(param.min, param.max)
 //             ),
 //             SpeedMS(
 //                 glm::linearRand(param.minSpeed, param.maxSpeed),
 //                 glm::linearRand(param.minSpeed, param.maxSpeed),
 //                 glm::linearRand(param.minSpeed, param.maxSpeed)
 //             ),
 //             AccemMS2(0., 0., 0.),
 //             glm::linearRand(param.minMass, param.maxMass)
 //             });
 //         if (i % 10000000 == 0) std::cout << i << std::endl;
 //     }
 //     stars.push_back(Star{
 //         PosM(
 //             0.f,
 //             0.f,
 //             0.f
 //         ),
 //         SpeedMS(
 //             0.f,
 //             0.f,
 //             0.f
 //         ),
 //         AccemMS2(
 //             0,
 //             0,
 //             0
 //         ),
 //         param.maxMass * 1e4
 //         });
 //     return stars;
 // };


// std::vector<Star> UniversGenerator::getCircularUniversPlummer(const size_t& nbStars, const GenParam& param)
// {
//     std::vector<Star> stars;
//     size_t N = nbStars;         // number of stars
//     double M = MILKYWAY_MASS;          // total mass of the galaxy
//     double R = param.Max;          // scale radius of the galaxy
//     double V = 0.9;          // scale velocity of the galaxy
// 
//     double starMass = M / N;
//     double blackHoleMass = starMass * 1e5;
//     for (size_t i{ 0u }; i < nbStars; ++i)
//     {
//         // https://stackoverflow.com/questions/28434765/how-to-randomly-distribute-n-masses-such-that-they-follow-a-plummer-density-dis
//         double a = 1. / 12.;
//         double mM = glm::linearRand(0.1, 0.99); // Proportion de mass entre l'etoile et le centre de la galaxy
//         double radius = R * a / glm::sqrt(glm::pow(mM, -2.0 / 3.0) - 1);
// 
//         double theta = glm::linearRand(0., 2.01 * glm::pi<double>());
// 
//         double massInside = mM * M + blackHoleMass;
//         double rSpeed = V * glm::sqrt(G * massInside / radius);
// 
//         stars.push_back(Star(
//             glm::linearRand(starMass * 0.0001, starMass * 1.9999),
//             glm::dvec3(
//                 radius * glm::cos(theta),
//                 radius * glm::sin(theta),
//                 0
//             ),
//             glm::dvec3(
//                 -rSpeed * glm::sin(theta),
//                 rSpeed * glm::cos(theta),
//                 0
//             )
//         ));
//     }
// 
//     stars.push_back(Star(
//         blackHoleMass,
//         glm::dvec3(
//             0,
//             0,
//             0
//         ),
//         glm::dvec3(
//             0,
//             0,
//             0
//         )
//     ));
//     return stars;
// }
#pragma once
#include "Physics/constant.hpp"
#include "Physics/Star.hpp"
#include <cuda_runtime.h>
#include <algorithm>

__device__ __host__ __inline__ Vec3 computeForce(const PosLy& pos1, MassMs mass1, const PosLy& pos2, MassMs mass2, DistanceLy d)
{
	return REDUCED_Gf * mass1* mass2 / (d * d * d + softening2) * (pos2 - pos1);
}

__device__ __host__ __inline__ Vec3 computeForce(const Star& s1, const Star& s2)
{
	DistanceLy d = Vec3::distance(s1.position, s2.position);
	return computeForce(s1.position, s1.mass, s2.position, s2.mass, d);
}


__device__ __host__ __inline__ float computeInvertGamma(const SpeedLs& speed)
{
	return std::sqrt(1.0f - fmin(.9999f, Vec3::dot(speed, speed)));
}

__device__ __host__ __inline__ Vec3 getAcceleration(const Star& s, const Vec3& force)
{
	// https://en.wikipedia.org/wiki/Relativistic_mechanics
	return computeInvertGamma(s.speed) / s.mass * (force - Vec3::dot(s.speed, force) * s.speed);
}

__device__ __host__ __inline__ void leapFrog(Star& s, const AccelLyY2& acc, const TimeY dt)
{
	s.speed += s.acceleration * dt / 2.f;

	// L'étoile la plus rapide de l'univers a une vitesse inferieure a quelque pourcent de la vitesse de la lumière. On limite donc la vitesse a 1% de la vitesse de la lumiere
	if (Vec3::length(s.speed) > 0.01f) 
	{
		s.speed = Vec3::normalize(s.speed) * 0.01f;
	}
	s.position += s.speed * dt;
	s.speed += acc * dt / 2.f;
	s.acceleration = acc;
}
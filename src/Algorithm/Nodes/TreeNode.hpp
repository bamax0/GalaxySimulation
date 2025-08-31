#pragma once
#include "LastTreeNode.hpp"
// #include "CudaLastTreeNode.hpp"
#include <array>

template<unsigned char N>
class GALAXY_SIM_DLL_EXPORT SimplifiedTreeNode : public ITreeNode
{
public:
	static float s_theta;

	SimplifiedTreeNode(const Bbox& bbox);
	~SimplifiedTreeNode();
	virtual void reset(const Bbox& bbox) override;

	virtual void startInserting() override {};
	virtual void endInserting() override {};
	virtual void appendStar(const Star& star, size_t depth = 0) override;
	virtual Vec3 computeTotalForce(const Star& star) const override;

	ITreeNode* getChild(unsigned char i);
	const ITreeNode* getChild(unsigned char i) const;
	void createAllChilds();
	virtual unsigned char getIndex(const PosLy& pos) const = 0;
protected:
	virtual void update(const PosLy& centerMass, MassMs mass) = 0;
	virtual PosLy getCenterOfMass() const = 0;
	virtual MassMs getMass() const = 0;

	virtual Bbox getSubBbox(unsigned char id) const = 0;
	virtual ITreeNode* createLastTreeNode(const PosLy& pos, MassMs mass, unsigned char) const
	{
		return new LastTreeNode(Star{ pos, Vec3{}, Vec3{}, mass });
	}
	virtual SimplifiedTreeNode<N>* createChild(const Bbox& bbox, unsigned char idx) const = 0;
	virtual SimplifiedTreeNode<N>* createChild(const Bbox& bbox, const PosLy& pos, MassMs mass, unsigned char idx) const
	{
		SimplifiedTreeNode<N>* child = createChild(bbox, idx);
		child->appendStar(Star{ pos, SpeedLs(), AccelLyY2(), mass });
		return child;
	}

	std::array<ITreeNode*, N> m_childs{ nullptr };
	Bbox m_bbox;
	size_t m_nbStar;

	static constexpr DistanceLy k_minBboxSize{ 8.f };
};

template<unsigned char N>
inline SimplifiedTreeNode<N>::SimplifiedTreeNode(const Bbox& bbox)
	: ITreeNode(), m_nbStar(0u), m_bbox(bbox)
{}

template<unsigned char N>
inline SimplifiedTreeNode<N>::~SimplifiedTreeNode()
{
	m_nbStar = 0;
	for (unsigned char i{ 0u }; i < N; ++i)
	{
		if (m_childs.at(i) != nullptr)
		{
			delete m_childs.at(i);
			m_childs.at(i) = nullptr;
		}
	}
}

template<unsigned char N>
void SimplifiedTreeNode<N>::reset(const Bbox& bbox)
{
	m_bbox = bbox;
	update({}, 0);
	m_nbStar = 0;
	for (unsigned char i{ 0u }; i < N; ++i)
	{
		if (m_childs.at(i) != nullptr)
		{
			delete m_childs.at(i);
			m_childs.at(i) = nullptr;
		}
	}
}

template<unsigned char N>
inline void SimplifiedTreeNode<N>::appendStar(const Star& star, size_t depth)
{
	++m_nbStar;
	if (m_nbStar == 1u) // Si c'est la première étoile
	{
		update(star.position, star.mass);
		return;
	}

	if (m_nbStar == 2u) // Si c'est la deuxième étoile
	{
		unsigned char idx0 = getIndex(getCenterOfMass());
		if (m_bbox.size < k_minBboxSize)
		{
			m_childs.at(idx0) = createLastTreeNode(getCenterOfMass(), getMass(), idx0);
		}
		else {
			// Il n'y avais aucun enfant avant, on peux donc créer le moeud sans vérification
			auto c = createChild(getSubBbox(idx0), getCenterOfMass(), getMass(), idx0);
			m_childs.at(idx0) = c;
		}
	}
	float newMass = getMass() + star.mass;
	update((getCenterOfMass() * getMass() + star.position * star.mass) / newMass, newMass);

	size_t idx = getIndex(star.position);
	ITreeNode* child = m_childs.at(idx);
	if (child == nullptr)
	{
		if (m_bbox.size < k_minBboxSize)
		{
			m_childs.at(idx) = createLastTreeNode(star.position, star.mass, idx);
			return;
		}
		Bbox b = getSubBbox(idx);
		//Solution pas belle
		m_childs.at(idx) = createChild(b, star.position, star.mass, idx);
		return;
	}
	child->appendStar(star, depth+1);
}

template<unsigned char N>
void SimplifiedTreeNode<N>::createAllChilds()
{
	for (unsigned short i{ 0u }; i < N; ++i)
	{
		if (m_childs.at(i) == nullptr)
		{
			SimplifiedTreeNode<N>* tn = createChild(getSubBbox(i), i);
			m_childs.at(i) = tn;
			tn->m_nbStar = 0u;
		}
	}
}

template<unsigned char N>
Vec3 SimplifiedTreeNode<N>::computeTotalForce(const Star& star) const
{
	if (getCenterOfMass() == star.position)
		return  { 0, 0, 0 };

	DistanceLy r = Vec3::distance(getCenterOfMass(), star.position);
	if (m_nbStar == 1 || m_bbox.size * 2.f < s_theta * r)
	{
		return computeForce(star.position, star.mass, getCenterOfMass(), getMass(), r);
	}

	Vec3 force{ 0, 0, 0 };

	for (const ITreeNode* child : m_childs)
	{
		if (child)
		{
			force += child->computeTotalForce(star);
		}
	}
	return force;
}

template<unsigned char N>
ITreeNode* SimplifiedTreeNode<N>::getChild(unsigned char i)
{
	return m_childs.at(i);
}

template<unsigned char N>
const ITreeNode* SimplifiedTreeNode<N>::getChild(unsigned char i) const
{
	return m_childs.at(i);
}

template<unsigned char N>
class GALAXY_SIM_DLL_EXPORT TreeNode : public SimplifiedTreeNode<N>
{
public:
	TreeNode(const Bbox& bbox);
	TreeNode(const Bbox& bbox, const PosLy& pos, MassMs mass);
	~TreeNode() = default;

	using SimplifiedTreeNode<N>::reset;
	using SimplifiedTreeNode<N>::startInserting;
	using SimplifiedTreeNode<N>::endInserting;
	using SimplifiedTreeNode<N>::appendStar;
	using SimplifiedTreeNode<N>::computeTotalForce;
	using SimplifiedTreeNode<N>::getIndex;

protected:
	virtual void update(const Vec3& centerMass, MassMs mass) override
	{
		m_centerMass = centerMass;
		m_mass = mass;
	}

	virtual PosLy getCenterOfMass() const override { return m_centerMass; }
	virtual MassMs getMass() const override { return m_mass; }

	PosLy m_centerMass;
	MassMs m_mass;

	using SimplifiedTreeNode<N>::getSubBbox;
	using SimplifiedTreeNode<N>::createLastTreeNode;
	using SimplifiedTreeNode<N>::createChild;
};

template<unsigned char N>
float SimplifiedTreeNode<N>::s_theta = 0.5f; // Valeur par défaut

template<unsigned char N>
TreeNode<N>::TreeNode(const Bbox& bbox) 
	: SimplifiedTreeNode<N>(bbox), m_centerMass({0.f, 0.f, 0.f}), m_mass(0) {}


template<unsigned char N>
TreeNode<N>::TreeNode(const Bbox& bbox, const PosLy& pos, MassMs mass)
	: SimplifiedTreeNode<N>(bbox), m_centerMass(pos), m_mass(mass)
{
	++this->m_nbStar;
}


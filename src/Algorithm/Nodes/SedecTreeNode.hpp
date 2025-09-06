#pragma once
#include "TreeNode.hpp"

/* Tres inefficace, utilisé octreeNode*/
class SedecTreeNode : public TreeNode<64>
{
public:
	SedecTreeNode(const Bbox& bbox) : TreeNode<64>(bbox) {}
	SedecTreeNode(const Bbox& bbox, const PosLy& pos, MassMs mass) : TreeNode<64>(bbox, pos, mass) {}
	virtual ~SedecTreeNode() override = default;
protected:

	virtual unsigned char getIndex(const PosLy& pos) const override
	{
		unsigned char idx{ 0u };
		float dx{pos.x - m_bbox.center.x};
		float dy{pos.y - m_bbox.center.y};
		float dz{pos.z - m_bbox.center.z};
		float halfSize{m_bbox.size / 2.f};

		if (dx > 0.f)
		{
			idx |= 1;
		}
		
		if (dy > 0.f)
		{
			idx |= 2;
		}
		
		if (dz > 0.f)
		{
			idx |= 4;
		}

		if (abs(dx) > halfSize)
		{
			idx |= 8;
		}
		
		if (abs(dy) > halfSize)
		{
			idx |= 16;
		}
		
		if (abs(dz) > halfSize)
		{
			idx |= 32;
		}
		return idx;
	}

	virtual Bbox getSubBbox(unsigned char id) const override
	{
		/*
		Ces conditions sont faite pour reflêter la méthode getIndex(); Les conditons peuvent paraitre complexe, mais ceci est fait pour
		conserver une méthode 'getIndex' le plus optimisé possible (la méthode getSubBbox() est moins appelé que getIndex, il est donc moins important
		d'optimisé celle-ci.

		Pour détailler le fonctionnement, voici l'explication de la première ligne:
		Les 4 valeurs possible de possition x du centre (relative au centre actuel)  sont: 0.75, 0.25, -0.25, -0.75  de la taille de la boite.
		Le première bit indique si le centre est >= 0.25 ou <= -0.25. Ainsi, si le 1er bit vaut 1 (c-a-d id & 1 == true), alors les 2 valeurs possible sont 0.25 et 0.75
		si le 4ème bit indique si on est abd(x) > 0.5, ainsi, si le 4ème bit vaut 1 (c-a-d id & 8 == true), alors les 2 valeurs possible sont 0.75 et -0.75

		Le calcul ci-dessous permet d'obtenir ce résultat.
		*/
		return Bbox{
			{m_bbox.center.x + m_bbox.size * ( ((id & 1) ? 0.5f : -0.5f)  * ((id & 8 ) ? 1.5f : 0.5f) ),
			 m_bbox.center.y + m_bbox.size * ( ((id & 2) ? 0.5f : -0.5f)  * ((id & 16) ? 1.5f : 0.5f) ),
			 m_bbox.center.z + m_bbox.size * ( ((id & 4) ? 0.5f : -0.5f)  * ((id & 32) ? 1.5f : 0.5f) )},
		m_bbox.size / 4.f
		};
	}

	virtual TreeNode<64>* createChild(const Bbox& bbox, unsigned char) const override
	{
		return new SedecTreeNode(bbox);
	}
	
	using TreeNode<64>::update;
	using TreeNode<64>::getCenterOfMass;
	using TreeNode<64>::getMass;

	using TreeNode<64>::reset;
	using TreeNode<64>::startInserting;
	using TreeNode<64>::endInserting;
	using TreeNode<64>::appendStar;
	using TreeNode<64>::computeTotalForce;
	using TreeNode<64>::createLastTreeNode;
	using TreeNode<64>::createChild;
};
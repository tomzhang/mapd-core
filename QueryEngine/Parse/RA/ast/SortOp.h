/**
 * @file	SortOp.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef RA_SORTOP_NODE_H
#define RA_SORTOP_NODE_H

#include <cassert>
#include "UnaryOp.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class SortOp : public UnaryOp {
    
public:
	RelExpr *n1 = NULL;
	AttrList *n2 = NULL;

	/// Constructor
	SortOp(RelExpr *n1, AttrList *n2) {
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;
	}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}

};

} // RA_Namespace

#endif // RA_SORTOP_NODE_H

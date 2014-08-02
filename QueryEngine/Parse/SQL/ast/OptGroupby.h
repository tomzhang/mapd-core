#ifndef SQL_OPTGROUPBY_H
#define SQL_OPTGROUPBY_H

#include <cassert>
#include "ASTNode.h"

namespace SQL_Namespace {

class OptGroupby : public ASTNode {

public:
	ColumnList* n1 = NULL;

	explicit OptGroupby(ColumnList* n1) {
		assert(n1);
		this->n1 = n1;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}

};

} // SQL_Namespace

#endif // SQL_OPTGROUPBY_H

#ifndef SQL_COLUMNLIST_H
#define SQL_COLUMNLIST_H

#include <cassert>
#include "ASTNode.h"

class ColumnList : public ASTNode {

public:

	ColumnList *n1 = NULL;
	Column *n2 = NULL;

	ColumnList(ColumnList *n1, Column *n2) {
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;
	}

	explicit ColumnList(Column *n2) {
		assert(n2);
		this->n2 = n2;
	}

	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}
};

#endif // SQL_COLUMNLIST_H

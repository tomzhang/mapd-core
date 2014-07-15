#ifndef AGGR_LIST_NODE_H
#define AGGR_LIST_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {
	class AggrList : public RelAlgNode {
    
public:

	AggrExpr* agex;
	AggrList* agLi;

	explicit AggrList(AggrList *n1, AggrExpr *n2) : agLi(n1), agex(n2) {}
	AggrList(AggrExpr *n) : agex(n), agLi(NULL) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // AGGR_LIST_NODE_H

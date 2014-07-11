#ifndef PROJECT_OP_NODE_H
#define PROJECT_OP_NODE_H

#include "RelAlgNode.h"
#include "UnaryOp.h"
#include "../visitor/Visitor.h"

class ProjectOp : public UnaryOp {
    
public:

	AttrList* atLi;

	explicit ProjectOp(RelExpr *n1, AttrList* n2) : atLi(n2) { relex = n1; }

/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
};

#endif // PROJECT_OP_NODE_H
#ifndef UNION_OP_NODE_H
#define UNION_OP_NODE_H

#include "RelAlgNode.h"
#include "BinaryOp.h"
#include "../visitor/Visitor.h"

class UnionOp : public BinaryOp {
    
public:

	explicit UnionOp(RelExpr *n1, RelExpr *n2) { relex1 = n1; relex2 = n2; }

/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
};

#endif // UNION_OP_NODE_H
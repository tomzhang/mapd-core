#ifndef SQL_AST_AMMSC_H
#define SQL_AST_AMMSC_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {

class  Ammsc : public ASTNode {
    
public:
    std::string funcName;
    
    /**< Constructor */
    explicit Ammsc(const std::string &n1) : funcName(n1) {}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

}

#endif // SQL_AST_AMMSC_H

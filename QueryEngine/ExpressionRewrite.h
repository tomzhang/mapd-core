#ifndef QUERYENGINE_EXPRESSIONREWRITE_H
#define QUERYENGINE_EXPRESSIONREWRITE_H

#include <memory>

namespace Analyzer {

class Expr;

class InValues;

}  // namespace Analyzer

// Rewrites an OR tree where leaves are equality compare against literals.
std::shared_ptr<Analyzer::Expr> rewrite_expr(const Analyzer::Expr*);

#endif  // QUERYENGINE_EXPRESSIONREWRITE_H

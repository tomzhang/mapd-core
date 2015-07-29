/**
 * @file    ParserNode.h
 * @author  Wei Hong <wei@map-d.com>
 * @brief   Classes representing a parse tree
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef PARSER_NODE_H_
#define PARSER_NODE_H_

#include <list>
#include <string>
#include <cstring>
#include <cstdint>
#include "../Shared/sqltypes.h"
#include "../Shared/sqldefs.h"
#include "../Analyzer/Analyzer.h"
#include "../Catalog/Catalog.h"

namespace Parser {

/*
 * @type Node
 * @brief root super class for all nodes in a pre-Analyzer
 * parse tree.
 */
class Node {
 public:
  virtual ~Node() {}
};

/*
 * @type SQLType
 * @brief class that captures type, predication and scale.
 */
class SQLType : public Node {
 public:
  explicit SQLType(SQLTypes t) : type(t), param1(-1), param2(0), is_array(false) {}
  SQLType(SQLTypes t, int p1) : type(t), param1(p1), param2(0), is_array(false) {}
  SQLType(SQLTypes t, int p1, int p2, bool a) : type(t), param1(p1), param2(p2), is_array(a) {}
  SQLTypes get_type() const { return type; }
  int get_param1() const { return param1; }
  int get_param2() const { return param2; }
  bool get_is_array() const { return is_array; }
  void set_is_array(bool a) { is_array = a; }
  std::string to_string() const;
  void check_type();

 private:
  SQLTypes type;
  int param1;  // e.g. for NUMERIC(10).  -1 means unspecified.
  int param2;  // e.g. for NUMERIC(10,3). 0 is default value.
  bool is_array;
};

/*
 * @type Expr
 * @brief root super class for all expression nodes
 */
class Expr : public Node {
 public:
  enum TlistRefType { TLIST_NONE, TLIST_REF, TLIST_COPY };
  /*
   * @brief Performs semantic analysis on the expression node
   * @param catalog The catalog object for the current database
   * @return An Analyzer::Expr object for the expression post semantic analysis
   */
  virtual std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const = 0;
  virtual std::string to_string() const = 0;
};

/*
 * @type Literal
 * @brief root super class for all literals
 */
class Literal : public Expr {
 public:
  virtual std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const = 0;
  virtual std::string to_string() const = 0;
};

/*
 * @type NullLiteral
 * @brief the Literal NULL
 */
class NullLiteral : public Literal {
 public:
  NullLiteral() {}
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const { return "NULL"; }
};

/*
 * @type StringLiteral
 * @brief the literal for string constants
 */
class StringLiteral : public Literal {
 public:
  explicit StringLiteral(std::string* s) : stringval(s) {}
  const std::string* get_stringval() const { return stringval.get(); }
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const { return "'" + *stringval + "'"; }

 private:
  std::unique_ptr<std::string> stringval;
};

/*
 * @type IntLiteral
 * @brief the literal for integer constants
 */
class IntLiteral : public Literal {
 public:
  explicit IntLiteral(int64_t i) : intval(i) {}
  int64_t get_intval() const { return intval; }
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const { return boost::lexical_cast<std::string>(intval); }

 private:
  int64_t intval;
};

/*
 * @type FixedPtLiteral
 * @brief the literal for DECIMAL and NUMERIC
 */
class FixedPtLiteral : public Literal {
 public:
  explicit FixedPtLiteral(std::string* n) : fixedptval(n) {}
  const std::string* get_fixedptval() const { return fixedptval.get(); }
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const { return *fixedptval; }

 private:
  std::unique_ptr<std::string> fixedptval;
};

/*
 * @type FloatLiteral
 * @brief the literal for FLOAT or REAL
 */
class FloatLiteral : public Literal {
 public:
  explicit FloatLiteral(float f) : floatval(f) {}
  float get_floatval() const { return floatval; }
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const { return boost::lexical_cast<std::string>(floatval); }

 private:
  float floatval;
};

/*
 * @type DoubleLiteral
 * @brief the literal for DOUBLE PRECISION
 */
class DoubleLiteral : public Literal {
 public:
  explicit DoubleLiteral(double d) : doubleval(d) {}
  double get_doubleval() const { return doubleval; }
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const { return boost::lexical_cast<std::string>(doubleval); }

 private:
  double doubleval;
};

/*
 * @type UserLiteral
 * @brief the literal for USER
 */
class UserLiteral : public Literal {
 public:
  UserLiteral() {}
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const { return "USER"; }
};

/*
 * @type OperExpr
 * @brief all operator expressions
 */
class OperExpr : public Expr {
 public:
  OperExpr(SQLOps t, Expr* l, Expr* r) : optype(t), opqualifier(kONE), left(l), right(r) {}
  OperExpr(SQLOps t, SQLQualifier q, Expr* l, Expr* r)
      : optype(t), opqualifier(q), left(l), right(r) {}
  SQLOps get_optype() const { return optype; }
  const Expr* get_left() const { return left.get(); }
  const Expr* get_right() const { return right.get(); }
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const;

 private:
  SQLOps optype;
  SQLQualifier opqualifier;
  std::unique_ptr<Expr> left;
  std::unique_ptr<Expr> right;
};

// forward reference of QuerySpec
class QuerySpec;

/*
 * @type SubqueryExpr
 * @brief expression for subquery
 */
class SubqueryExpr : public Expr {
 public:
  explicit SubqueryExpr(QuerySpec* q) : query(q) {}
  const QuerySpec* get_query() const { return query.get(); }
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const;

 private:
  std::unique_ptr<QuerySpec> query;
};

class IsNullExpr : public Expr {
 public:
  IsNullExpr(bool n, Expr* a) : is_not(n), arg(a) {}
  bool get_is_not() const { return is_not; }
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const;

 private:
  bool is_not;
  std::unique_ptr<Expr> arg;
};

/*
 * @type InExpr
 * @brief expression for the IS NULL predicate
 */
class InExpr : public Expr {
 public:
  InExpr(bool n, Expr* a) : is_not(n), arg(a) {}
  bool get_is_not() const { return is_not; }
  const Expr* get_arg() const { return arg.get(); }
  virtual std::shared_ptr<Analyzer::Expr> analyze(
      const Catalog_Namespace::Catalog& catalog,
      Analyzer::Query& query,
      TlistRefType allow_tlist_ref = TLIST_NONE) const = 0;
  virtual std::string to_string() const;

 protected:
  bool is_not;
  std::unique_ptr<Expr> arg;
};

/*
 * @type InSubquery
 * @brief expression for the IN (subquery) predicate
 */
class InSubquery : public InExpr {
 public:
  InSubquery(bool n, Expr* a, SubqueryExpr* q) : InExpr(n, a), subquery(q) {}
  const SubqueryExpr* get_subquery() const { return subquery.get(); }
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const;

 private:
  std::unique_ptr<SubqueryExpr> subquery;
};

/*
 * @type InValues
 * @brief expression for IN (val1, val2, ...)
 */
class InValues : public InExpr {
 public:
  InValues(bool n, Expr* a, std::list<Expr*>* v) : InExpr(n, a) {
    CHECK(v);
    for (const auto e : *v) {
      value_list.emplace_back(e);
    }
    delete v;
  }
  const std::list<std::unique_ptr<Expr>>& get_value_list() const { return value_list; }
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const;

 private:
  std::list<std::unique_ptr<Expr>> value_list;
};

/*
 * @type BetweenExpr
 * @brief expression for BETWEEN lower AND upper
 */
class BetweenExpr : public Expr {
 public:
  BetweenExpr(bool n, Expr* a, Expr* l, Expr* u) : is_not(n), arg(a), lower(l), upper(u) {}
  bool get_is_not() const { return is_not; }
  const Expr* get_arg() const { return arg.get(); }
  const Expr* get_lower() const { return lower.get(); }
  const Expr* get_upper() const { return upper.get(); }
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const;

 private:
  bool is_not;
  std::unique_ptr<Expr> arg;
  std::unique_ptr<Expr> lower;
  std::unique_ptr<Expr> upper;
};

/*
 * @type LikeExpr
 * @brief expression for the LIKE predicate
 */
class LikeExpr : public Expr {
 public:
  LikeExpr(bool n, bool i, Expr* a, Expr* l, Expr* e)
      : is_not(n), is_ilike(i), arg(a), like_string(l), escape_string(e) {}
  bool get_is_not() const { return is_not; }
  const Expr* get_arg() const { return arg.get(); }
  const Expr* get_like_string() const { return like_string.get(); }
  const Expr* get_escape_string() const { return escape_string.get(); }
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const;

 private:
  bool is_not;
  bool is_ilike;
  std::unique_ptr<Expr> arg;
  std::unique_ptr<Expr> like_string;
  std::unique_ptr<Expr> escape_string;

  static void check_like_expr(const std::string& like_str, char escape_char);
  static bool test_is_simple_expr(const std::string& like_str, char escape_char);
  static void erase_cntl_chars(std::string& like_str, char escape_char);
};

/*
 * @type ExistsExpr
 * @brief expression for EXISTS (subquery)
 */
class ExistsExpr : public Expr {
 public:
  explicit ExistsExpr(QuerySpec* q) : query(q) {}
  const QuerySpec* get_query() const { return query.get(); }
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const;

 private:
  std::unique_ptr<QuerySpec> query;
};

/*
 * @type ColumnRef
 * @brief expression for a column reference
 */
class ColumnRef : public Expr {
 public:
  explicit ColumnRef(std::string* n1) : table(nullptr), column(n1) {}
  ColumnRef(std::string* n1, std::string* n2) : table(n1), column(n2) {}
  const std::string* get_table() const { return table.get(); }
  const std::string* get_column() const { return column.get(); }
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const;

 private:
  std::unique_ptr<std::string> table;
  std::unique_ptr<std::string> column;  // can be nullptr in the t.* case
};

/*
 * @type FunctionRef
 * @brief expression for a function call
 */
class FunctionRef : public Expr {
 public:
  explicit FunctionRef(std::string* n) : name(n), distinct(false), arg(nullptr) {}
  FunctionRef(std::string* n, Expr* a) : name(n), distinct(false), arg(a) {}
  FunctionRef(std::string* n, bool d, Expr* a) : name(n), distinct(d), arg(a) {}
  const std::string* get_name() const { return name.get(); }
  bool get_distinct() const { return distinct; }
  Expr* get_arg() const { return arg.get(); }
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const;

 private:
  std::unique_ptr<std::string> name;
  bool distinct;              // only true for COUNT(DISTINCT x)
  std::unique_ptr<Expr> arg;  // for COUNT, nullptr means '*'
};

class CastExpr : public Expr {
 public:
  CastExpr(Expr* a, SQLType* t) : arg(a), target_type(t) {}
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const {
    return "CAST(" + arg->to_string() + " AS " + target_type->to_string() + ")";
  }

 private:
  std::unique_ptr<Expr> arg;
  std::unique_ptr<SQLType> target_type;
};

class ExprPair : public Node {
 public:
  ExprPair(Expr* e1, Expr* e2) : expr1(e1), expr2(e2) {}
  const Expr* get_expr1() const { return expr1.get(); }
  const Expr* get_expr2() const { return expr2.get(); }

 private:
  std::unique_ptr<Expr> expr1;
  std::unique_ptr<Expr> expr2;
};

class CaseExpr : public Expr {
 public:
  CaseExpr(std::list<ExprPair*>* w, Expr* e) : else_expr(e) {
    CHECK(w);
    for (const auto e : *w) {
      when_then_list.emplace_back(e);
    }
    delete w;
  }
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const;

 private:
  std::list<std::unique_ptr<ExprPair>> when_then_list;
  std::unique_ptr<Expr> else_expr;
};

class ExtractExpr : public Expr {
 public:
  ExtractExpr(std::string* f, Expr* a) : field(f), from_arg(a) {}
  virtual std::shared_ptr<Analyzer::Expr> analyze(const Catalog_Namespace::Catalog& catalog,
                                                  Analyzer::Query& query,
                                                  TlistRefType allow_tlist_ref = TLIST_NONE) const;
  virtual std::string to_string() const;

 private:
  std::unique_ptr<std::string> field;
  std::unique_ptr<Expr> from_arg;
};

/*
 * @type TableRef
 * @brief table reference in FROM clause
 */
class TableRef : public Node {
 public:
  explicit TableRef(std::string* t) : table_name(t), range_var(nullptr) {}
  TableRef(std::string* t, std::string* r) : table_name(t), range_var(r) {}
  const std::string* get_table_name() const { return table_name.get(); }
  const std::string* get_range_var() const { return range_var.get(); }
  std::string to_string() const;

 private:
  std::unique_ptr<std::string> table_name;
  std::unique_ptr<std::string> range_var;
};

/*
 * @type Stmt
 * @brief root super class for all SQL statements
 */
class Stmt : public Node {
  // intentionally empty
};

/*
 * @type DMLStmt
 * @brief DML Statements
 */
class DMLStmt : public Stmt {
 public:
  virtual void analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const = 0;
};

/*
 * @type DDLStmt
 * @brief DDL Statements
 */
class DDLStmt : public Stmt {
 public:
  virtual void execute(const Catalog_Namespace::SessionInfo& session) = 0;
};

/*
 * @type TableElement
 * @brief elements in table definition
 */
class TableElement : public Node {
  // intentionally empty
};

/*
 * @type ColumnConstraintDef
 * @brief integrity constraint on a column
 */
class ColumnConstraintDef : public Node {
 public:
  ColumnConstraintDef(bool n, bool u, bool p, Literal* d)
      : notnull(n), unique(u), is_primarykey(p), defaultval(d) {}
  ColumnConstraintDef(Expr* c)
      : notnull(false), unique(false), is_primarykey(false), check_condition(c) {}
  ColumnConstraintDef(std::string* t, std::string* c)
      : notnull(false), unique(false), is_primarykey(false), foreign_table(t), foreign_column(c) {}
  bool get_notnull() const { return notnull; }
  bool get_unique() const { return unique; }
  bool get_is_primarykey() const { return is_primarykey; }
  const Literal* get_defaultval() const { return defaultval.get(); }
  const Expr* get_check_condition() const { return check_condition.get(); }
  const std::string* get_foreign_table() const { return foreign_table.get(); }
  const std::string* get_foreign_column() const { return foreign_column.get(); }

 private:
  bool notnull;
  bool unique;
  bool is_primarykey;
  std::unique_ptr<Literal> defaultval;
  std::unique_ptr<Expr> check_condition;
  std::unique_ptr<std::string> foreign_table;
  std::unique_ptr<std::string> foreign_column;
};

/*
 * @type CompressDef
 * @brief Node for compression scheme definition
 */
class CompressDef : public Node {
 public:
  CompressDef(std::string* n, int p) : encoding_name(n), encoding_param(p) {}
  const std::string* get_encoding_name() const { return encoding_name.get(); }
  int get_encoding_param() const { return encoding_param; }

 private:
  std::unique_ptr<std::string> encoding_name;
  int encoding_param;
};

/*
 * @type ColumnDef
 * @brief Column definition
 */
class ColumnDef : public TableElement {
 public:
  ColumnDef(std::string* c, SQLType* t, CompressDef* cp, ColumnConstraintDef* cc)
      : column_name(c), column_type(t), compression(cp), column_constraint(cc) {}
  const std::string* get_column_name() const { return column_name.get(); }
  SQLType* get_column_type() const { return column_type.get(); }
  const CompressDef* get_compression() const { return compression.get(); }
  const ColumnConstraintDef* get_column_constraint() const { return column_constraint.get(); }

 private:
  std::unique_ptr<std::string> column_name;
  std::unique_ptr<SQLType> column_type;
  std::unique_ptr<CompressDef> compression;
  std::unique_ptr<ColumnConstraintDef> column_constraint;
};

/*
 * @type TableConstraintDef
 * @brief integrity constraint for table
 */
class TableConstraintDef : public TableElement {
  // intentionally empty
};

/*
 * @type UniqueDef
 * @brief uniqueness constraint
 */
class UniqueDef : public TableConstraintDef {
 public:
  UniqueDef(bool p, std::list<std::string*>* cl) : is_primarykey(p) {
    CHECK(cl);
    for (const auto s : *cl) {
      column_list.emplace_back(s);
    }
    delete cl;
  }
  bool get_is_primarykey() const { return is_primarykey; }
  const std::list<std::unique_ptr<std::string>>& get_column_list() const { return column_list; }

 private:
  bool is_primarykey;
  std::list<std::unique_ptr<std::string>> column_list;
};

/*
 * @type ForeignKeyDef
 * @brief foreign key constraint
 */
class ForeignKeyDef : public TableConstraintDef {
 public:
  ForeignKeyDef(std::list<std::string*>* cl, std::string* t, std::list<std::string*>* fcl)
      : foreign_table(t) {
    CHECK(cl);
    for (const auto s : *cl) {
      column_list.emplace_back(s);
    }
    delete cl;
    if (fcl) {
      for (const auto s : *fcl) {
        foreign_column_list.emplace_back(s);
      }
    }
    delete fcl;
  }
  const std::list<std::unique_ptr<std::string>>& get_column_list() const { return column_list; }
  const std::string* get_foreign_table() const { return foreign_table.get(); }
  const std::list<std::unique_ptr<std::string>>& get_foreign_column_list() const {
    return foreign_column_list;
  }

 private:
  std::list<std::unique_ptr<std::string>> column_list;
  std::unique_ptr<std::string> foreign_table;
  std::list<std::unique_ptr<std::string>> foreign_column_list;
};

/*
 * @type CheckDef
 * @brief Check constraint
 */
class CheckDef : public TableConstraintDef {
 public:
  CheckDef(Expr* c) : check_condition(c) {}
  const Expr* get_check_condition() const { return check_condition.get(); }

 private:
  std::unique_ptr<Expr> check_condition;
};

/*
 * @type NameValueAssign
 * @brief Assignment of a string value to a named attribute
 */
class NameValueAssign : public Node {
 public:
  NameValueAssign(std::string* n, Literal* v) : name(n), value(v) {}
  const std::string* get_name() const { return name.get(); }
  const Literal* get_value() const { return value.get(); }

 private:
  std::unique_ptr<std::string> name;
  std::unique_ptr<Literal> value;
};

/*
 * @type CreateTableStmt
 * @brief CREATE TABLE statement
 */
class CreateTableStmt : public DDLStmt {
 public:
  CreateTableStmt(std::string* tab,
                  std::list<TableElement*>* table_elems,
                  bool i,
                  std::list<NameValueAssign*>* s)
      : table(tab), if_not_exists(i) {
    CHECK(table_elems);
    for (const auto e : *table_elems) {
      table_element_list.emplace_back(e);
    }
    delete table_elems;
    if (s) {
      for (const auto e : *s) {
        storage_options.emplace_back(e);
      }
      delete s;
    }
  }
  const std::string* get_table() const { return table.get(); }
  const std::list<std::unique_ptr<TableElement>>& get_table_element_list() const {
    return table_element_list;
  }

  virtual void execute(const Catalog_Namespace::SessionInfo& session);

 private:
  std::unique_ptr<std::string> table;
  std::list<std::unique_ptr<TableElement>> table_element_list;
  bool if_not_exists;
  std::list<std::unique_ptr<NameValueAssign>> storage_options;
};

/*
 * @type DropTableStmt
 * @brief DROP TABLE statement
 */
class DropTableStmt : public DDLStmt {
 public:
  DropTableStmt(std::string* tab, bool i) : table(tab), if_exists(i) {}
  const std::string* get_table() const { return table.get(); }
  virtual void execute(const Catalog_Namespace::SessionInfo& session);

 private:
  std::unique_ptr<std::string> table;
  bool if_exists;
};

/*
 * @type CopyTableStmt
 * @brief COPY ... FROM ...
 */
class CopyTableStmt : public DDLStmt {
 public:
  CopyTableStmt(std::string* t, std::string* f, std::list<NameValueAssign*>* o)
      : table(t), file_path(f) {
    if (o) {
      for (const auto e : *o) {
        options.emplace_back(e);
      }
      delete o;
    }
  }
  virtual void execute(const Catalog_Namespace::SessionInfo& session);

 private:
  std::unique_ptr<std::string> table;
  std::unique_ptr<std::string> file_path;
  std::list<std::unique_ptr<NameValueAssign>> options;
};

/*
 * @type QueryExpr
 * @brief query expression
 */
class QueryExpr : public Node {
 public:
  virtual void analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const = 0;
};

/*
 * @type UnionQuery
 * @brief UNION or UNION ALL queries
 */
class UnionQuery : public QueryExpr {
 public:
  UnionQuery(bool u, QueryExpr* l, QueryExpr* r) : is_unionall(u), left(l), right(r) {}
  bool get_is_unionall() const { return is_unionall; }
  const QueryExpr* get_left() const { return left.get(); }
  const QueryExpr* get_right() const { return right.get(); }
  virtual void analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const;

 private:
  bool is_unionall;
  std::unique_ptr<QueryExpr> left;
  std::unique_ptr<QueryExpr> right;
};

class SelectEntry : public Node {
 public:
  SelectEntry(Expr* e, std::string* r) : select_expr(e), alias(r) {}
  const Expr* get_select_expr() const { return select_expr.get(); }
  const std::string* get_alias() const { return alias.get(); }
  std::string to_string() const;

 private:
  std::unique_ptr<Expr> select_expr;
  std::unique_ptr<std::string> alias;
};

/*
 * @type QuerySpec
 * @brief a simple query
 */
class QuerySpec : public QueryExpr {
 public:
  QuerySpec(bool d,
            std::list<SelectEntry*>* s,
            std::list<TableRef*>* f,
            Expr* w,
            std::list<Expr*>* g,
            Expr* h)
      : is_distinct(d), where_clause(w), having_clause(h) {
    if (s) {
      for (const auto e : *s) {
        select_clause.emplace_back(e);
      }
      delete s;
    }
    CHECK(f);
    for (const auto e : *f) {
      from_clause.emplace_back(e);
    }
    delete f;
    if (g) {
      for (const auto e : *g) {
        groupby_clause.emplace_back(e);
      }
      delete g;
    }
  }
  bool get_is_distinct() const { return is_distinct; }
  const std::list<std::unique_ptr<SelectEntry>>& get_select_clause() const { return select_clause; }
  const std::list<std::unique_ptr<TableRef>>& get_from_clause() const { return from_clause; }
  const Expr* get_where_clause() const { return where_clause.get(); }
  const std::list<std::unique_ptr<Expr>>& get_groupby_clause() const { return groupby_clause; }
  const Expr* get_having_clause() const { return having_clause.get(); }
  virtual void analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const;
  std::string to_string() const;

 private:
  bool is_distinct;
  std::list<std::unique_ptr<SelectEntry>> select_clause; /* nullptr means SELECT * */
  std::list<std::unique_ptr<TableRef>> from_clause;
  std::unique_ptr<Expr> where_clause;
  std::list<std::unique_ptr<Expr>> groupby_clause;
  std::unique_ptr<Expr> having_clause;
  void analyze_from_clause(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const;
  void analyze_select_clause(const Catalog_Namespace::Catalog& catalog,
                             Analyzer::Query& query) const;
  void analyze_where_clause(const Catalog_Namespace::Catalog& catalog,
                            Analyzer::Query& query) const;
  void analyze_group_by(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const;
  void analyze_having_clause(const Catalog_Namespace::Catalog& catalog,
                             Analyzer::Query& query) const;
};

/*
 * @type OrderSpec
 * @brief order spec for a column in ORDER BY clause
 */
class OrderSpec : public Node {
 public:
  OrderSpec(int n, ColumnRef* c, bool d, bool f)
      : colno(n), column(c), is_desc(d), nulls_first(f) {}
  int get_colno() const { return colno; }
  const ColumnRef* get_column() const { return column.get(); }
  bool get_is_desc() const { return is_desc; }
  bool get_nulls_first() const { return nulls_first; }

 private:
  int colno; /* 0 means use column name */
  std::unique_ptr<ColumnRef> column;
  bool is_desc;
  bool nulls_first;
};

/*
 * @type SelectStmt
 * @brief SELECT statement
 */
class SelectStmt : public DMLStmt {
 public:
  SelectStmt(QueryExpr* q, std::list<OrderSpec*>* o, int64_t l, int64_t f)
      : query_expr(q), limit(l), offset(f) {
    if (o) {
      for (const auto e : *o) {
        orderby_clause.emplace_back(e);
      }
      delete o;
    }
  }
  const QueryExpr* get_query_expr() const { return query_expr.get(); }
  const std::list<std::unique_ptr<OrderSpec>>& get_orderby_clause() const { return orderby_clause; }
  virtual void analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const;

 private:
  std::unique_ptr<QueryExpr> query_expr;
  std::list<std::unique_ptr<OrderSpec>> orderby_clause;
  int64_t limit;
  int64_t offset;
};

/*
 * @type ExplainStmt
 * @brief EXPLAIN DMLStmt
 */
class ExplainStmt : public DDLStmt {
 public:
  ExplainStmt(DMLStmt* s) : stmt(s) {}
  const DMLStmt* get_stmt() const { return stmt.get(); }
  virtual void execute(const Catalog_Namespace::SessionInfo& session) { CHECK(false); }

 private:
  std::unique_ptr<DMLStmt> stmt;
};

/*
 * @type ExportQueryStmt
 * @brief COPY ( query ) TO file ...
 */
class ExportQueryStmt : public DDLStmt {
 public:
  ExportQueryStmt(SelectStmt* q, std::string* p, std::list<NameValueAssign*>* o)
      : select_stmt(q), file_path(p) {
    if (o) {
      for (const auto e : *o) {
        options.emplace_back(e);
      }
      delete o;
    }
  }
  virtual void execute(const Catalog_Namespace::SessionInfo& session);

 private:
  std::unique_ptr<SelectStmt> select_stmt;
  std::unique_ptr<std::string> file_path;
  std::list<std::unique_ptr<NameValueAssign>> options;
};

/*
 * @type CreateViewStmt
 * @brief CREATE VIEW statement
 */
class CreateViewStmt : public DDLStmt {
 public:
  CreateViewStmt(std::string* v,
                 std::list<std::string*>* c,
                 QuerySpec* q,
                 bool ck,
                 bool m,
                 std::list<NameValueAssign*>* o,
                 bool i)
      : view_name(v), query(q), checkoption(ck), is_materialized(m), if_not_exists(i) {
    if (c) {
      for (const auto e : *c) {
        column_list.emplace_back(e);
      }
      delete c;
    }
    if (o) {
      for (const auto e : *o) {
        matview_options.emplace_back(e);
      }
      delete o;
    }
  }
  const std::string* get_view_name() const { return view_name.get(); }
  const std::list<std::unique_ptr<std::string>>& get_column_list() const { return column_list; }
  const QuerySpec* get_query() const { return query.get(); }
  bool get_checkoption() const { return checkoption; }
  virtual void execute(const Catalog_Namespace::SessionInfo& session);

 private:
  std::unique_ptr<std::string> view_name;
  std::list<std::unique_ptr<std::string>> column_list;
  std::unique_ptr<QuerySpec> query;
  bool checkoption;
  bool is_materialized;
  std::list<std::unique_ptr<NameValueAssign>> matview_options;
  bool if_not_exists;
};

class RefreshViewStmt : public DDLStmt {
 public:
  explicit RefreshViewStmt(std::string* v) : view_name(v) {}
  virtual void execute(const Catalog_Namespace::SessionInfo& session);

 private:
  std::unique_ptr<std::string> view_name;
};

/*
 * @type DropViewStmt
 * @brief DROP VIEW statement
 */
class DropViewStmt : public DDLStmt {
 public:
  DropViewStmt(std::string* v, bool i) : view_name(v), if_exists(i) {}
  const std::string* get_view_name() const { return view_name.get(); }
  virtual void execute(const Catalog_Namespace::SessionInfo& session);

 private:
  std::unique_ptr<std::string> view_name;
  bool if_exists;
};

/*
 * @type CreateDBStmt
 * @brief CREATE DATABASE statement
 */
class CreateDBStmt : public DDLStmt {
 public:
  CreateDBStmt(std::string* n, std::list<NameValueAssign*>* l) : db_name(n) {
    if (l) {
      for (const auto e : *l) {
        name_value_list.emplace_back(e);
      }
      delete l;
    }
  }
  virtual void execute(const Catalog_Namespace::SessionInfo& session);

 private:
  std::unique_ptr<std::string> db_name;
  std::list<std::unique_ptr<NameValueAssign>> name_value_list;
};

/*
 * @type DropDBStmt
 * @brief DROP DATABASE statement
 */
class DropDBStmt : public DDLStmt {
 public:
  explicit DropDBStmt(std::string* n) : db_name(n) {}
  virtual void execute(const Catalog_Namespace::SessionInfo& session);

 private:
  std::unique_ptr<std::string> db_name;
};

/*
 * @type CreateUserStmt
 * @brief CREATE USER statement
 */
class CreateUserStmt : public DDLStmt {
 public:
  CreateUserStmt(std::string* n, std::list<NameValueAssign*>* l) : user_name(n) {
    if (l) {
      for (const auto e : *l) {
        name_value_list.emplace_back(e);
      }
      delete l;
    }
  }
  virtual void execute(const Catalog_Namespace::SessionInfo& session);

 private:
  std::unique_ptr<std::string> user_name;
  std::list<std::unique_ptr<NameValueAssign>> name_value_list;
};

/*
 * @type AlterUserStmt
 * @brief ALTER USER statement
 */
class AlterUserStmt : public DDLStmt {
 public:
  AlterUserStmt(std::string* n, std::list<NameValueAssign*>* l) : user_name(n) {
    if (l) {
      for (const auto e : *l) {
        name_value_list.emplace_back(e);
      }
      delete l;
    }
  }
  virtual void execute(const Catalog_Namespace::SessionInfo& session);

 private:
  std::unique_ptr<std::string> user_name;
  std::list<std::unique_ptr<NameValueAssign>> name_value_list;
};

/*
 * @type DropUserStmt
 * @brief DROP USER statement
 */
class DropUserStmt : public DDLStmt {
 public:
  DropUserStmt(std::string* n) : user_name(n) {}
  virtual void execute(const Catalog_Namespace::SessionInfo& session);

 private:
  std::unique_ptr<std::string> user_name;
};

/*
 * @type InsertStmt
 * @brief super class for INSERT statements
 */
class InsertStmt : public DMLStmt {
 public:
  InsertStmt(std::string* t, std::list<std::string*>* c) : table(t) {
    if (c) {
      for (const auto e : *c) {
        column_list.emplace_back(e);
      }
      delete c;
    }
  }
  const std::string* get_table() const { return table.get(); }
  const std::list<std::unique_ptr<std::string>>& get_column_list() const { return column_list; }
  virtual void analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const = 0;

 private:
  std::unique_ptr<std::string> table;
  std::list<std::unique_ptr<std::string>> column_list;
};

/*
 * @type InsertValuesStmt
 * @brief INSERT INTO ... VALUES ...
 */
class InsertValuesStmt : public InsertStmt {
 public:
  InsertValuesStmt(std::string* t, std::list<std::string*>* c, std::list<Expr*>* v)
      : InsertStmt(t, c) {
    CHECK(v);
    for (const auto e : *v) {
      value_list.emplace_back(e);
    }
    delete v;
  }
  const std::list<std::unique_ptr<Expr>>& get_value_list() const { return value_list; }
  virtual void analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const;

 private:
  std::list<std::unique_ptr<Expr>> value_list;
};

/*
 * @type InsertQueryStmt
 * @brief INSERT INTO ... SELECT ...
 */
class InsertQueryStmt : public InsertStmt {
 public:
  InsertQueryStmt(std::string* t, std::list<std::string*>* c, QuerySpec* q)
      : InsertStmt(t, c), query(q) {}
  const QuerySpec* get_query() const { return query.get(); }
  virtual void analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const;

 private:
  std::unique_ptr<QuerySpec> query;
};

/*
 * @type Assignment
 * @brief assignment in UPDATE statement
 */
class Assignment : public Node {
 public:
  Assignment(std::string* c, Expr* a) : column(c), assignment(a) {}
  const std::string* get_column() const { return column.get(); }
  const Expr* get_assignment() const { return assignment.get(); }

 private:
  std::unique_ptr<std::string> column;
  std::unique_ptr<Expr> assignment;
};

/*
 * @type UpdateStmt
 * @brief UPDATE statement
 */
class UpdateStmt : public DMLStmt {
 public:
  UpdateStmt(std::string* t, std::list<Assignment*>* a, Expr* w) : table(t), where_clause(w) {
    CHECK(a);
    for (const auto e : *a) {
      assignment_list.emplace_back(e);
    }
    delete a;
  }
  const std::string* get_table() const { return table.get(); }
  const std::list<std::unique_ptr<Assignment>>& get_assignment_list() const {
    return assignment_list;
  }
  const Expr* get_where_clause() const { return where_clause.get(); }
  virtual void analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const;

 private:
  std::unique_ptr<std::string> table;
  std::list<std::unique_ptr<Assignment>> assignment_list;
  std::unique_ptr<Expr> where_clause;
};

/*
 * @type DeleteStmt
 * @brief DELETE statement
 */
class DeleteStmt : public DMLStmt {
 public:
  DeleteStmt(std::string* t, Expr* w) : table(t), where_clause(w) {}
  const std::string* get_table() const { return table.get(); }
  const Expr* get_where_clause() const { return where_clause.get(); }
  virtual void analyze(const Catalog_Namespace::Catalog& catalog, Analyzer::Query& query) const;

 private:
  std::unique_ptr<std::string> table;
  std::unique_ptr<Expr> where_clause;
};
}
#endif  // PARSERNODE_H_

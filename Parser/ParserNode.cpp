/**
 * @file		ParserNode.cpp
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Functions for ParserNode classes
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <cassert>
#include <stdexcept>
#include <typeinfo>
#include <boost/algorithm/string/predicate.hpp>
#include "../Catalog/Catalog.h"
#include "ParserNode.h"

namespace Parser {
	SubqueryExpr::~SubqueryExpr() 
	{
		delete query;
	}

	ExistsExpr::~ExistsExpr() 
	{
		delete query;
	}

	InValues::~InValues() 
	{
		for (auto p : *value_list)
			delete p;
	}

	BetweenExpr::~BetweenExpr() 
	{
		delete arg;
		delete lower;
		delete upper;
	}

	LikeExpr::~LikeExpr() 
	{
		delete arg;
		delete like_string;
		if (escape_string != nullptr)
			delete escape_string;
	}

	ColumnRef::~ColumnRef() 
	{
		if (table != nullptr)
			delete table;
		if (column != nullptr)
			delete column;
	}
	
	TableRef::~TableRef() 
	{
		delete table_name;
		if (range_var != nullptr)
			delete range_var;
	}

	ColumnConstraintDef::~ColumnConstraintDef() 
	{
		if (defaultval != nullptr)
			delete defaultval;
		if (check_condition != nullptr)
			delete check_condition;
		if (foreign_table != nullptr)
			delete foreign_table;
		if (foreign_column != nullptr)
			delete foreign_column;
	}

	ColumnDef::~ColumnDef() 
	{
		delete column_name;
		delete column_type;
		if (column_constraint != nullptr)
			delete column_constraint;
	}

	UniqueDef::~UniqueDef() 
	{
		for (auto p : *column_list)
			delete p;
	}

	ForeignKeyDef::~ForeignKeyDef() 
	{
		for (auto p : *column_list)
			delete p;
		delete foreign_table;
		if (foreign_column_list != nullptr) {
			for (auto p : *foreign_column_list)
				delete p;
		}
	}

	FunctionRef::~FunctionRef() 
	{
		delete name;
		if (arg != nullptr)
			delete arg;
	}

	QuerySpec::~QuerySpec() 
	{
		delete select_clause;
		delete from_clause;
		if (where_clause != nullptr)
			delete where_clause;
		if (groupby_clause != nullptr)
			delete groupby_clause;
		if (having_clause != nullptr)
			delete having_clause;
	}

	Analyzer::Expr *
	NullLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		Analyzer::Constant *c = new Analyzer::Constant(kNULLT, true);
		Datum d;
		d.pointerval = nullptr;
		c->set_constval(d);
		return c;
	}
	
	Analyzer::Expr *
	StringLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		SQLTypeInfo ti;
		ti.type = kVARCHAR;
		ti.dimension = stringval->length();
		ti.scale = 0;
		Analyzer::Constant *c = new Analyzer::Constant(ti, false);
		char *s = new char[stringval->length() + 1];
		strcpy(s, stringval->c_str());
		Datum d;
		d.pointerval = (void*)s;
		c->set_constval(d);
		return c;
	}

	Analyzer::Expr *
	IntLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		SQLTypes t;
		Datum d;
		if (intval >= INT16_MIN && intval <= INT16_MAX) {
			t = kSMALLINT;
			d.smallintval = (int16_t)intval;
		} else if (intval >= INT32_MIN && intval <= INT32_MAX) {
			t = kINT;
			d.intval = (int32_t)intval;
		} else {
			t = kBIGINT;
			d.bigintval = intval;
		}
		Analyzer::Constant *c = new Analyzer::Constant(t, false);
		c->set_constval(d);
		return c;
	}

	Analyzer::Expr *
	FixedPtLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		assert(fixedptval->length() <= 20);
		size_t dot = fixedptval->find_first_of('.', 0);
		assert(dot != std::string::npos);
		std::string before_dot = fixedptval->substr(0, dot);
		std::string after_dot = fixedptval->substr(dot+1);
		Datum d;
		d.bigintval = std::stoll(before_dot);
		int64_t fraction = std::stoll(after_dot);
		SQLTypeInfo ti;
		ti.type = kNUMERIC;
		ti.scale = after_dot.length();
		ti.dimension = before_dot.length() + ti.scale;
		// the following loop can be made more efficient if needed
		for (int i = 0; i < ti.scale; i++)
			d.bigintval *= 10;
		d.bigintval += fraction;
		if (ti.dimension < 11)
			d.intval = (int)d.bigintval;
		return new Analyzer::Constant(ti, false, d);
	}

	Analyzer::Expr *
	FloatLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		Datum d;
		d.floatval = floatval;
		return new Analyzer::Constant(kFLOAT, false, d);
	}
	
	Analyzer::Expr *
	DoubleLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		Datum d;
		d.doubleval = doubleval;
		return new Analyzer::Constant(kDOUBLE, false, d);
	}

	Analyzer::Expr *
	UserLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		throw std::runtime_error("USER literal not supported yet.");
		return nullptr;
	}

	Analyzer::Expr *
	OperExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		SQLTypeInfo result_type, left_type, right_type;
		SQLTypeInfo new_left_type, new_right_type;
		Analyzer::Expr *left_expr, *right_expr;
		SQLQualifier qual = kONE;
		if (typeid(*right) == typeid(SubqueryExpr))
			qual = dynamic_cast<SubqueryExpr*>(right)->get_qualifier();
		left_expr = left->analyze(catalog, query);
		right_expr = right->analyze(catalog, query);
		left_type = left_expr->get_type_info();
		right_type = right_expr->get_type_info();
		result_type = Analyzer::BinOper::analyze_type_info(optype, left_type, right_type, &new_left_type, &new_right_type);
		if (left_type != new_left_type)
			left_expr = left_expr->add_cast(new_left_type);
		if (right_type != new_right_type)
			right_expr = right_expr->add_cast(new_right_type);
		return new Analyzer::BinOper(result_type, optype, qual, left_expr, right_expr);
	}

	Analyzer::Expr *
	SubqueryExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		throw std::runtime_error("Subqueries are not supported yet.");
		return nullptr;
	}

	Analyzer::Expr *
	IsNullExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		Analyzer::Expr *arg_expr = arg->analyze(catalog, query);
		Analyzer::Expr *result = new Analyzer::UOper(kBOOLEAN, kISNULL, arg_expr);
		if (is_not)
			result = new Analyzer::UOper(kBOOLEAN, kNOT, result);
		return result;
	}

	Analyzer::Expr *
	InSubquery::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		throw std::runtime_error("Subqueries are not supported yet.");
		return nullptr;
	}

	Analyzer::Expr *
	InValues::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		Analyzer::Expr *arg_expr = arg->analyze(catalog, query);
		std::list<Analyzer::Expr*> *value_exprs = new std::list<Analyzer::Expr*>();
		for (auto p : *value_list) {
			Analyzer::Expr *e = p->analyze(catalog, query);
			value_exprs->push_back(e->add_cast(arg_expr->get_type_info()));
		}
		Analyzer::Expr *result = new Analyzer::InValues(arg_expr, value_exprs);
		if (is_not)
			result = new Analyzer::UOper(kBOOLEAN, kNOT, result);
		return result;
	}

	Analyzer::Expr *
	BetweenExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		Analyzer::Expr *arg_expr = arg->analyze(catalog, query);
		Analyzer::Expr *lower_expr = lower->analyze(catalog, query);
		Analyzer::Expr *upper_expr = upper->analyze(catalog, query);
		SQLTypeInfo new_left_type, new_right_type;
		(void)Analyzer::BinOper::analyze_type_info(kGE, arg_expr->get_type_info(), lower_expr->get_type_info(), &new_left_type, &new_right_type);
		Analyzer::BinOper *lower_pred = new Analyzer::BinOper(kBOOLEAN, kGE, kONE, arg_expr->add_cast(new_left_type), lower_expr->add_cast(new_right_type));
		(void)Analyzer::BinOper::analyze_type_info(kLE, arg_expr->get_type_info(), lower_expr->get_type_info(), &new_left_type, &new_right_type);
		Analyzer::BinOper *upper_pred = new Analyzer::BinOper(kBOOLEAN, kLE, kONE, arg_expr->add_cast(new_left_type), upper_expr->add_cast(new_right_type));
		Analyzer::Expr *result = new Analyzer::BinOper(kBOOLEAN, kAND, kONE, lower_pred, upper_pred);
		if (is_not)
			result = new Analyzer::UOper(kBOOLEAN, kNOT, result);
		return result;
	}

	Analyzer::Expr *
	LikeExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		Analyzer::Expr *arg_expr = arg->analyze(catalog, query);
		Analyzer::Expr *like_expr = like_string->analyze(catalog, query);
		Analyzer::Expr *escape_expr = escape_string == nullptr ? nullptr: escape_string->analyze(catalog, query);
		if (!IS_STRING(arg_expr->get_type_info().type))
			throw std::runtime_error("expression before LIKE must be of a string type.");
		if (!IS_STRING(like_expr->get_type_info().type))
			throw std::runtime_error("expression after LIKE must be of a string type.");
		if (escape_expr != nullptr && !IS_STRING(escape_expr->get_type_info().type))
			throw std::runtime_error("expression after ESCAPE must be of a string type.");
		Analyzer::Expr *result = new Analyzer::LikeExpr(arg_expr, like_expr, escape_expr);
		if (is_not)
			result = new Analyzer::UOper(kBOOLEAN, kNOT, result);
		return result;
	}

	Analyzer::Expr *
	ExistsExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		throw std::runtime_error("Subqueries are not supported yet.");
		return nullptr;
	}

	Analyzer::Expr *
	ColumnRef::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		int table_id;
		const ColumnDescriptor *cd;
		if (column == nullptr)
			throw std::runtime_error("invalid column name *.");
		if (table != nullptr) {
			Analyzer::RangeTblEntry *rte = query.get_rte(*table);
			if (rte == nullptr)
				throw std::runtime_error("range variable or table name " + *table + " does not exist.");
			cd = rte->get_column_desc(catalog, *column);
			if (cd == nullptr)
				throw std::runtime_error("Column name " + *column + " does not exist.");
			table_id = rte->get_table_id();
		} else {
			bool found = false;
			for (auto rte : *query.get_rangetable()) {
				cd = rte->get_column_desc(catalog, *column);
				if (cd != nullptr && !found) {
					found = true;
					table_id = rte->get_table_id();
				}
				if (cd != nullptr && found)
					throw std::runtime_error("Column name " + *column + " is ambiguous.");
			}
			if (cd == nullptr)
				throw std::runtime_error("Column name " + *column + " does not exist.");
		}
		return new Analyzer::ColumnVar(cd->columnType, table_id, cd->columnId);
	}

	Analyzer::Expr *
	FunctionRef::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		SQLTypeInfo result_type;
		SQLAgg agg_type;
		Analyzer::Expr *arg_expr;
		bool is_distinct = false;
		if (boost::iequals(*name, "count")) {
			result_type.type = kINT;
			if (arg == nullptr)
				arg_expr = nullptr;
			else
				arg_expr = arg->analyze(catalog, query);
			is_distinct = distinct;
		}
		else if (boost::iequals(*name, "min")) {
			agg_type = kMIN;
			arg_expr = arg->analyze(catalog, query);
			result_type = arg_expr->get_type_info();
		}
		else if (boost::iequals(*name, "max")) {
			agg_type = kMAX;
			arg_expr = arg->analyze(catalog, query);
			result_type = arg_expr->get_type_info();
		}
		else if (boost::iequals(*name, "avg")) {
			agg_type = kAVG;
			arg_expr = arg->analyze(catalog, query);
			result_type = arg_expr->get_type_info();
		}
		else if (boost::iequals(*name, "sum")) {
			agg_type = kSUM;
			arg_expr = arg->analyze(catalog, query);
			result_type = arg_expr->get_type_info();
		}
		else
			throw std::runtime_error("invalid function name: " + *name);
		return new Analyzer::AggExpr(result_type, agg_type, arg_expr, is_distinct);
	}

	void
	UnionQuery::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		left->analyze(catalog, query);
		Analyzer::Query *right_query = new Analyzer::Query();
		right->analyze(catalog, *right_query);
		query.set_next_query(right_query);
		query.set_is_unionall(is_unionall);
	}

	void
	QuerySpec::analyze_having_clause(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		if (having_clause == nullptr) {
			query.set_having_predicate(nullptr);
			return;
		}
		Analyzer::Expr *p = having_clause->analyze(catalog, query);
		if (p->get_type_info().type != kBOOLEAN)
			throw std::runtime_error("Only boolean expressions can be in HAVING clause.");
		p->check_group_by(query.get_group_by());
		query.set_having_predicate(p);
	}

	void
	QuerySpec::analyze_group_by(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		if (groupby_clause == nullptr) {
			query.set_group_by(nullptr);
			return;
		}
		std::list<Analyzer::Expr*> *groupby = new std::list<Analyzer::Expr*>();
		for (auto c : *groupby_clause) {
			Analyzer::Expr *e = c->analyze(catalog, query);
			groupby->push_back(e);
		}
		for (auto t : *query.get_targetlist()) {
			Analyzer::Expr *e = t->get_expr();
			if (typeid(*e) != typeid(Analyzer::AggExpr))
				e->check_group_by(groupby);
		}
		query.set_group_by(groupby);
	}

	void
	QuerySpec::analyze_where_clause(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		if (where_clause == nullptr) {
			query.set_where_predicate(nullptr);
			return;
		}
		Analyzer::Expr *p = where_clause->analyze(catalog, query);
		if (p->get_type_info().type != kBOOLEAN)
			throw std::runtime_error("Only boolean expressions can be in WHERE clause.");
		query.set_where_predicate(p);
	}

	void
	QuerySpec::analyze_select_clause(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		std::vector<Analyzer::TargetEntry*> *tlist = new std::vector<Analyzer::TargetEntry*>();
		if (select_clause == nullptr) {
			// this means SELECT *
			for (auto rte : *query.get_rangetable()) {
				rte->expand_star_in_targetlist(catalog, rte->get_table_id(), *tlist);
			}
		}
		else {
			std::string resname;
			for (auto p : *select_clause) {
				const Parser::Expr *select_expr = p->get_select_expr();
				// look for the case of range_var.*
				if (typeid(*select_expr) == typeid(ColumnRef) &&
						dynamic_cast<const ColumnRef*>(select_expr)->get_column() == nullptr) {
						const std::string *range_var_name = dynamic_cast<const ColumnRef*>(select_expr)->get_table();
						Analyzer::RangeTblEntry *rte = query.get_rte(*range_var_name);
						if (rte == nullptr)
							throw std::runtime_error("invalid range variable name: " + *range_var_name);
						rte->expand_star_in_targetlist(catalog, rte->get_table_id(), *tlist);
				}
				else {
					Analyzer::Expr *e = select_expr->analyze(catalog, query);

					if (p->get_alias() != nullptr)
						resname = *p->get_alias();
					else if (typeid(*e) == typeid(Analyzer::ColumnVar)) {
						Analyzer::ColumnVar *colvar = dynamic_cast<Analyzer::ColumnVar*>(e);
						const ColumnDescriptor *col_desc = catalog.getMetadataForColumn(colvar->get_table_id(), colvar->get_column_id());
						resname = col_desc->columnName;
					}
					Analyzer::TargetEntry *tle = new Analyzer::TargetEntry(tlist->size() + 1, resname, e);
					tlist->push_back(tle);
				}
			}
		}
		query.set_targetlist(tlist);
	}

	void
	QuerySpec::analyze_from_clause(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		Analyzer::RangeTblEntry *rte;
		std::vector<Analyzer::RangeTblEntry*> *rt = new std::vector<Analyzer::RangeTblEntry*>();
		for (auto p : *from_clause) {
			const TableDescriptor *table_desc;
			table_desc = catalog.getMetadataForTable(*p->get_table_name());
			if (table_desc == nullptr)
				throw std::runtime_error("table does not exist." + *p->get_table_name());
			std::string range_var;
			if (p->get_range_var() == nullptr)
				range_var = *p->get_table_name();
			else
				range_var = *p->get_range_var();
			rte = new Analyzer::RangeTblEntry(range_var, table_desc->tableId, *p->get_table_name(), nullptr);
			rt->push_back(rte);
		}
		query.set_rangetable(rt);
	}

	void
	QuerySpec::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		query.set_is_distinct(is_distinct);
		analyze_from_clause(catalog, query);
		analyze_select_clause(catalog, query);
		analyze_where_clause(catalog, query);
		analyze_group_by(catalog, query);
		analyze_having_clause(catalog, query);
	}

	void
	SelectStmt::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		query_expr->analyze(catalog, query);
		if (orderby_clause == nullptr) {
			query.set_order_by(nullptr);
			return;
		}
		const std::vector<Analyzer::TargetEntry*> *tlist;
		tlist = query.get_targetlist();
		std::list<Analyzer::OrderEntry> *order_by = new std::list<Analyzer::OrderEntry>();
		for (auto p : *orderby_clause) {
			int tle_no = p->get_colno();
			if (tle_no == 0) {
				// use column name
				// search through targetlist for matching name
				const std::string *name = p->get_column()->get_column();
				for (auto tle : *tlist) {
					if (tle->get_resname() == *name) {
						tle_no = tle->get_resno();
						break;
					}
				}
				if (tle_no == 0)
					throw std::runtime_error("invalid name in order by: " + *name);
			}
			order_by->push_back(Analyzer::OrderEntry(tle_no, p->get_is_desc(), p->get_nulls_first()));
		}
		query.set_order_by(order_by);
	}

	void
	UpdateStmt::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		throw std::runtime_error("UPDATE statement not supported yet.");
	}

	void
	DeleteStmt::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		throw std::runtime_error("DELETE statement not supported yet.");
	}

	void
	CreateTableStmt::execute(Catalog_Namespace::Catalog &catalog)
	{
		if (catalog.getMetadataForTable(*table) != nullptr)
			throw std::runtime_error("Table " + *table + " already exits.");
		std::vector<ColumnDescriptor *> columns;
		for (auto e : *table_element_list) {
			if (typeid(*e) != typeid(ColumnDef))
				throw std::runtime_error("Table constraints are not supported yet.");
			ColumnDef *coldef = dynamic_cast<ColumnDef*>(e);
			ColumnDescriptor *cd = new ColumnDescriptor();
			cd->columnName = *coldef->get_column_name();
			const SQLType *t = coldef->get_column_type();
			cd->columnType.type = t->get_type();
			cd->columnType.dimension = t->get_param1();
			cd->columnType.scale = t->get_param2();
			const ColumnConstraintDef *cc = coldef->get_column_constraint();
			if (cc == nullptr)
				cd->columnType.notnull = false;
			else {
				cd->columnType.notnull = cc->get_notnull();
			}
			columns.push_back(cd);
		}
		catalog.createTable(*table, columns);
	}

	void
	CreateViewStmt::execute(Catalog_Namespace::Catalog &catalog)
	{
		throw std::runtime_error("CREATE VIEW not supported yet.");
	}

}

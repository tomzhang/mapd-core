/**
 * @file		Planner.h
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Defines data structures for query plan nodes.
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef PLANNER_H
#define PLANNER_H

#include <cstdint>
#include <string>
#include <vector>
#include <list>
#include <map>
#include "../Analyzer/Analyzer.h"

// forward class references to avoid including Thrift headers
class TRenderProperty;
typedef std::map<std::string, TRenderProperty> TRenderPropertyMap;
typedef std::map<std::string, TRenderPropertyMap> TColumnRenderMap;

namespace Planner {
	/*
	 * @type Plan
	 * @brief super class for all plan nodes
	 */
	class Plan {
		public:
			Plan(const std::vector<Analyzer::TargetEntry*> &t, const std::list<std::shared_ptr<Analyzer::Expr>>& q, double c, Plan *p)
        : targetlist(t), quals(q), cost(c), child_plan(p) {}
			Plan(const std::vector<Analyzer::TargetEntry*> &t, double c, Plan *p) : targetlist(t), cost(c), child_plan(p) {}
			Plan() : cost(0.0), child_plan(nullptr) {}
			Plan(const std::vector<Analyzer::TargetEntry*> &t) : targetlist(t), cost(0.0), child_plan(nullptr) {}
			virtual ~Plan();
			const std::vector<Analyzer::TargetEntry*>& get_targetlist() const { return targetlist; }
			const std::list<std::shared_ptr<Analyzer::Expr>>& get_quals() const { return quals; }
			double get_cost() const { return cost; }
			const Plan *get_child_plan() const { return child_plan; }
			void add_tle(Analyzer::TargetEntry *tle) { targetlist.push_back(tle); }
			void set_targetlist(const std::vector<Analyzer::TargetEntry*> &t) { targetlist = t; }
			virtual void print() const;
		protected:
			std::vector<Analyzer::TargetEntry*> targetlist; // projection of this plan node
			std::list<std::shared_ptr<Analyzer::Expr>> quals; // list of boolean expressions, implicitly conjunctive
			double cost; // Planner assigned cost for optimization purpose
			Plan *child_plan; // most plan nodes have at least one child, therefore keep it in super class
	};

	/*
	 * @type Result
	 * @brief Result node is for evaluating constant predicates, e.g., 1 < 2 or $1 = 'foo'
	 * It is also used to perform further projections and qualifications 
	 * from the child_plan.  One use case is  to eliminate columns that 
	 * are required by the child_plan but not in the final targetlist
	 * , e.g., group by columns that are not selected.  Another use case is
	 * to evaluate expressions over group by columns and aggregates in the targetlist.
	 * A 3rd use case is to process the HAVING clause for filtering 
	 * rows produced by AggPlan.
	 */
	class Result : public Plan {
		public:
			Result(
        std::vector<Analyzer::TargetEntry*>& t,
        const std::list<std::shared_ptr<Analyzer::Expr>>& q,
        double c,
        Plan *p,
        const std::list<std::shared_ptr<Analyzer::Expr>>& cq) : Plan(t, q, c, p), const_quals(cq) {}
			const std::list<std::shared_ptr<Analyzer::Expr>>& get_constquals() const { return const_quals; }
			virtual void print() const;
		private:
			std::list<std::shared_ptr<Analyzer::Expr>> const_quals; // constant quals to evaluate only once
	};

	/*
	 * @type Scan
	 * @brief Scan node is for scanning a table or rowset.
	 */
	class Scan : public Plan {
		public:
			Scan(
          const std::vector<Analyzer::TargetEntry*>& t,
          const std::list<std::shared_ptr<Analyzer::Expr>>& q,
          double c,
          Plan *p,
          std::list<std::shared_ptr<Analyzer::Expr>>& sq,
          int r,
          const std::list<int>& cl)
        : Plan(t, q, c, p), simple_quals(sq), table_id(r), col_list(cl) {}
			Scan(const Analyzer::RangeTblEntry &rte);
			const std::list<std::shared_ptr<Analyzer::Expr>>& get_simple_quals() const { return simple_quals; };
			int get_table_id() const { return table_id; }
			const std::list<int> &get_col_list() const { return col_list; }
			void add_predicate(std::shared_ptr<Analyzer::Expr> pred) { quals.push_back(pred); }
			void add_simple_predicate(std::shared_ptr<Analyzer::Expr> pred) { simple_quals.push_back(pred); }
			virtual void print() const;
		private:
			// simple_quals consists of predicates of the form 'ColumnVar BinOper Constant'
			// it can be used for eliminating fragments and/or partitions from the scan.
			std::list<std::shared_ptr<Analyzer::Expr>> simple_quals;
			int table_id; // rangetable entry index for the table to scan
			std::list<int> col_list; // list of column ids to scan
	};

	/*
	 * @type ValuesScan
	 * @brief ValuesScan returns a row from a list of values.
	 * It is used for processing INSERT INTO tab VALUES (...)
	 */
	class ValuesScan : public Plan {
		public:
			ValuesScan(const std::vector<Analyzer::TargetEntry*> &t) : Plan(t) {}
			virtual ~ValuesScan() {};
			virtual void print() const;
	};

	/*
	 * @type Join
	 * @brief super class for all join nodes.
	 */
	class Join : public Plan {
		public:
			Join(
          const std::vector<Analyzer::TargetEntry*>& t,
          const std::list<std::shared_ptr<Analyzer::Expr>>& q,
          double c, Plan *p, Plan *cp2)
        : Plan(t, q, c, p), child_plan2(cp2) {}
			virtual ~Join();
			virtual void print() const;
			const Plan *get_outerplan() const { return child_plan; }
			const Plan *get_innerplan() const { return child_plan2; }
		private:
			Plan *child_plan2;
	};

	/*
	 * @type AggPlan
	 * @brief AggPlan handles aggregate functions and group by.
	 */
	class AggPlan : public Plan {
		public:
			AggPlan(
          const std::vector<Analyzer::TargetEntry*>& t,
          double c,
          Plan *p,
          const std::list<std::shared_ptr<Analyzer::Expr>>& gl) : Plan(t, c, p), groupby_list(gl) {}
			const std::list<std::shared_ptr<Analyzer::Expr>>& get_groupby_list() const { return groupby_list; }
			virtual void print() const;
		private:
			std::list<std::shared_ptr<Analyzer::Expr>> groupby_list; // list of expressions for group by.  only Var nodes are allow now.
	};

	/*
	 * @type Append
	 * @brief Append evaluates a list of query plans one by one and simply appends all rows
	 * to result set.  It is for processing UNION ALL queries.
	 */
	class Append : public Plan {
		public:
			Append(
          const std::vector<Analyzer::TargetEntry*>& t,
          const std::list<std::shared_ptr<Analyzer::Expr>>& q,
          double c,
          Plan *p,
          const std::list<Plan*> &pl) : Plan(t, q, c, p), plan_list(pl) {}
			virtual ~Append();
			const std::list<Plan*> &get_plan_list() const { return plan_list; }
			virtual void print() const;
		private:
			std::list<Plan*> plan_list; // list of plans to union all
	};

	/*
	 * @type MergeAppend
	 * @brief MergeAppend merges sorted streams of rows and eliminate duplicates.
	 * It is for processing UNION queries.
	 */
	class MergeAppend : public Plan {
		public:
			MergeAppend(
          const std::vector<Analyzer::TargetEntry*>& t,
          const std::list<std::shared_ptr<Analyzer::Expr>>& q,
          double c,
          Plan *p,
          const std::list<Plan*>& pl,
          const std::list<Analyzer::OrderEntry>& oe) : Plan(t, q, c, p), mergeplan_list(pl), order_entries(oe) {}
			virtual ~MergeAppend();
			const std::list<Plan*> &get_mergeplan_list() const { return mergeplan_list; }
			const std::list<Analyzer::OrderEntry> &get_order_entries() const { return order_entries; }
			virtual void print() const;
		private:
			std::list<Plan*> mergeplan_list; // list of plans to merge
			std::list<Analyzer::OrderEntry> order_entries; // defines how the mergeplans are sorted
	};

	/*
	 * @type Sort
	 * @brief Sort node returns rows from the child_plan sorted by selected columns.
	 * It handles the Order By clause and is used for mergejoin and for remove duplicates.
	 */
	class Sort : public Plan {
		public:
			Sort(const std::vector<Analyzer::TargetEntry*> &t, double c, Plan *p, const std::list<Analyzer::OrderEntry> &oe, bool d) : Plan(t, c, p), order_entries(oe), remove_duplicates(d) {}
			virtual ~Sort();
			const std::list<Analyzer::OrderEntry> &get_order_entries() const { return order_entries; }
			bool get_remove_duplicates() const { return remove_duplicates; }
			virtual void print() const;
		private:
			std::list<Analyzer::OrderEntry> order_entries; // defines columns to sort on and in what order
			bool remove_duplicates;
	};

	/*
	 * @type RootPlan
	 * @brief RootPlan is the end result produced by the Planner for the Executor to execute.
	 * @TODO add support for parametrized queries.
	 */
	class RootPlan {
		public:
      enum Dest {
        kCLIENT,
        kEXPLAIN,
        kRENDER
      };
			RootPlan(Plan *p, SQLStmtType t, int r, const std::list<int> &c, const Catalog_Namespace::Catalog &cat, int64_t l, int64_t o) : plan(p), stmt_type(t), result_table_id(r), result_col_list(c), catalog(cat), limit(l), offset(o), render_type("NONE"), render_properties(nullptr), column_render_properties(nullptr), plan_dest(kCLIENT) {}
			~RootPlan();
			const Plan *get_plan() const { return plan; }
			SQLStmtType get_stmt_type() const { return stmt_type; }
			int get_result_table_id() const { return result_table_id; }
			const std::list<int> &get_result_col_list() const { return result_col_list; }
			const Catalog_Namespace::Catalog &get_catalog() const { return catalog; }
			virtual void print() const;
			int64_t get_limit() const { return limit; }
			int64_t get_offset() const { return offset; }
      const std::string &get_render_type() const { return render_type; }
      void set_render_type(std::string t) { render_type = t; }
      const TRenderPropertyMap *get_render_properties() const { return render_properties; }
      void set_render_properties(const TRenderPropertyMap *r) { render_properties = r; }
      const TColumnRenderMap *get_column_render_properties() const { return column_render_properties; }
      void set_column_render_properties(const TColumnRenderMap *p) { column_render_properties = p; }
      Dest get_plan_dest() const { return plan_dest; }
      void set_plan_dest(Dest d) { plan_dest = d; }
		private:
			Plan *plan; // query plan
			SQLStmtType stmt_type; // SELECT, UPDATE, DELETE or INSERT
			int result_table_id; // For UPDATE, DELETE or INSERT only: table id for the result table
			std::list<int> result_col_list; // For UPDATE and INSERT only: list of result column ids.
			const Catalog_Namespace::Catalog &catalog; // include the current catalog here for the executor
			int64_t limit; // limit from LIMIT clause.  0 means ALL
			int64_t offset; // offset from OFFSET clause.  0 means no offset.
      std::string render_type;
      const TRenderPropertyMap *render_properties;
      const TColumnRenderMap *column_render_properties;
      Dest plan_dest;
	};

	/*
	 * @type Optimizer
	 * @brief This is the class for performing query optimizations.
	 */
	class Optimizer {
		public:
			Optimizer(const Analyzer::Query &q, const Catalog_Namespace::Catalog &c) : cur_query(nullptr), query(q), catalog(c) {}
			~Optimizer() {}
			/*
			 * @brief optimize optimize an entire SQL DML statement
			 */
			RootPlan *optimize();
		private:
			/*
			 * @brief optimize_query optimize the query portion of the statement.  can be a union
			 * query
			 */
			Plan *optimize_query();
			/*
			 * @brief optimize_current_query optimize cur_query and output plan to cur_plan.  
			 * must be a non-union query.
			 */
			void optimize_current_query();
			void optimize_scans();
			void optimize_joins();
			void optimize_aggs();
			void optimize_orderby();
			void process_targetlist();
			std::vector<Scan*> base_scans;
			std::list<const Analyzer::Expr*> join_predicates;
			std::list<const Analyzer::Expr*> const_predicates;
			const Analyzer::Query *cur_query;
			Plan *cur_plan;
			const Analyzer::Query &query;
			const Catalog_Namespace::Catalog &catalog;
	};
}

#endif // PLANNER_H

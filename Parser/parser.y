%name Parser
%define CLASS SQLParser
%define LVAL yylval
%define CONSTRUCTOR_INIT : lexer(yylval)
%define MEMBERS                 \
		virtual ~SQLParser() {} \
    int parse(const std::string & inputStr, std::list<Stmt*>& parseTrees, std::string &lastParsed) { std::istringstream ss(inputStr); lexer.switch_streams(&ss,0);  yyparse(parseTrees); lastParsed = lexer.YYText(); return yynerrs; } \
    private:                   \
       SQLLexer lexer;
%define LEX_BODY {return lexer.yylex();}
%define ERROR_BODY {} /*{ std::cerr << "Syntax error on line " << lexer.lineno() << ". Last word parsed: " << lexer.YYText() << std::endl; } */

%header{
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <list>
#include <string>
#include <utility>
#include <stdexcept>
#include <boost/algorithm/string/predicate.hpp>
#include <FlexLexer.h>
#include "ParserNode.h"

using namespace Parser;
#define YY_Parser_PARSE_PARAM std::list<Stmt*>& parseTrees
%}

%union {
	bool boolval;
	int64_t	intval;
	float floatval;
	double doubleval;
	std::string *stringval;
	SQLOps opval;
	SQLQualifier qualval;
	std::list<Node*> *listval;
	std::list<std::string*> *slistval;
	Node *nodeval;
}

%header{
	class SQLLexer : public yyFlexLexer {
		public:
			SQLLexer(YY_Parser_STYPE &lval) : yylval(lval) {};
			YY_Parser_STYPE &yylval;
	};
%}

	/* symbolic tokens */

%token NAME
%token STRING
%token INTNUM FIXEDNUM

	/* operators */

%left OR
%left AND
%left NOT
%left EQUAL COMPARISON /* = <> < > <= >= */
%left '+' '-'
%left '*' '/'
%nonassoc UMINUS

	/* literal keyword tokens */

%token ALL ALTER AMMSC ANY AS ASC AUTHORIZATION BETWEEN BIGINT BOOLEAN BY
%token CASE CAST CHARACTER CHECK CLOSE COMMIT CONTINUE CREATE CURRENT
%token DATABASE DATE CURSOR DECIMAL DECLARE DEFAULT DELETE DESC DISTINCT DOUBLE DROP
%token ELSE END ESCAPE EXISTS FETCH FIRST FLOAT FOR FOREIGN FOUND FROM 
%token GRANT GROUP HAVING IF IN INSERT INTEGER INTO
%token IS KEY LANGUAGE LAST LIKE LIMIT NULLX NUMERIC OF OFFSET ON OPEN OPTION
%token ORDER PARAMETER PRECISION PRIMARY PRIVILEGES PROCEDURE
%token PUBLIC REAL REFERENCES ROLLBACK SCHEMA SELECT SET
%token SMALLINT SOME TABLE TEXT THEN TIME TIMESTAMP TO UNION
%token UNIQUE UPDATE USER VALUES VIEW WHEN WHENEVER WHERE WITH WORK

%start sql_list

%%

sql_list:
		sql ';'	{ parseTrees.push_front(dynamic_cast<Stmt*>($<nodeval>1)); }
	|	sql_list sql ';' 
	{ 
		parseTrees.push_front(dynamic_cast<Stmt*>($<nodeval>2));
	}
	;


	/* schema definition language */
sql:		/* schema {	$<nodeval>$ = $<nodeval>1; } */
	 create_table_statement { $<nodeval>$ = $<nodeval>1; }
	| create_view_statement { $<nodeval>$ = $<nodeval>1; }
	/* | prvililege_def { $<nodeval>$ = $<nodeval>1; } */
	| drop_view_statement { $<nodeval>$ = $<nodeval>1; }
	| refresh_view_statement { $<nodeval>$ = $<nodeval>1; }
	| drop_table_statement { $<nodeval>$ = $<nodeval>1; }
	| create_database_statement { $<nodeval>$ = $<nodeval>1; }
	| drop_database_statement { $<nodeval>$ = $<nodeval>1; }
	| create_user_statement { $<nodeval>$ = $<nodeval>1; }
	| drop_user_statement { $<nodeval>$ = $<nodeval>1; }
	| alter_user_statement { $<nodeval>$ = $<nodeval>1; }
	;
	
/* NOT SUPPORTED
schema:
		CREATE SCHEMA AUTHORIZATION user opt_schema_element_list
	;

opt_schema_element_list:
		{ $<listval>$ = nullptr; } // empty
	|	schema_element_list {	$<listval>$ = $<listval>1; }
	;

schema_element_list:
		schema_element 
	|	schema_element_list schema_element
	;

schema_element:
		create_table_statement {	$$ = $1; }
	|	create_view_statement {	$$ = $1; }
	|	privilege_def {	$$ = $1; }
	;
NOT SUPPORTED */

create_database_statement:
		CREATE DATABASE NAME
		{
			$<nodeval>$ = new CreateDBStmt($<stringval>3, nullptr);
		}
		| CREATE DATABASE NAME '(' name_eq_value_list ')'
		{
			$<nodeval>$ = new CreateDBStmt($<stringval>3, reinterpret_cast<std::list<NameValueAssign*>*>($<listval>5));
		}
		;
drop_database_statement:
		DROP DATABASE NAME
		{
			$<nodeval>$ = new DropDBStmt($<stringval>3);
		}
		;
create_user_statement:
		CREATE USER NAME '(' name_eq_value_list ')'
		{
			$<nodeval>$ = new CreateUserStmt($<stringval>3, reinterpret_cast<std::list<NameValueAssign*>*>($<listval>5));
		}
		;
drop_user_statement:
		DROP USER NAME
		{
			$<nodeval>$ = new DropUserStmt($<stringval>3);
		}
		;
alter_user_statement:
		ALTER USER NAME '(' name_eq_value_list ')'
		{
			$<nodeval>$ = new AlterUserStmt($<stringval>3, reinterpret_cast<std::list<NameValueAssign*>*>($<listval>5));
		}
		;
name_eq_value_list:
		name_eq_value
		{
			$<listval>$ = new std::list<Node*>(1, $<nodeval>1);
		}
		| name_eq_value_list ',' name_eq_value
		{
			$<listval>$ = $<listval>1;
			$<listval>$->push_back($<nodeval>3);
		}
		;
name_eq_value: NAME EQUAL literal { $<nodeval>$ = new NameValueAssign($<stringval>1, dynamic_cast<Literal*>($<nodeval>3)); }
		;

opt_if_not_exists:
		IF NOT EXISTS { $<boolval>$ = true; }
		| /* empty */ { $<boolval>$ = false; }
		;

create_table_statement:
		CREATE TABLE opt_if_not_exists table '(' base_table_element_commalist ')' opt_with_option_list
		{
			$<nodeval>$ = new CreateTableStmt($<stringval>4, reinterpret_cast<std::list<TableElement*>*>($<listval>6), $<boolval>3, reinterpret_cast<std::list<NameValueAssign*>*>($<listval>8));
		}
	;

opt_if_exists:
		IF EXISTS { $<boolval>$ = true; }
		| /* empty */ { $<boolval>$ = false; }
		;

drop_table_statement:
		DROP TABLE opt_if_exists table
		{
			$<nodeval>$ = new DropTableStmt($<stringval>4, $<boolval>3);
		}
		;

base_table_element_commalist:
		base_table_element { $<listval>$ = new std::list<Node*>(1, $<nodeval>1); }
	|	base_table_element_commalist ',' base_table_element
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

base_table_element:
		column_def { $<nodeval>$ = $<nodeval>1; }
	|	table_constraint_def { $<nodeval>$ = $<nodeval>1; }
	;

column_def:
		column data_type opt_compression
		{	$<nodeval>$ = new ColumnDef($<stringval>1, dynamic_cast<SQLType*>($<nodeval>2), dynamic_cast<CompressDef*>($<nodeval>3), nullptr); }
		| column data_type column_constraint_def opt_compression 
		{ $<nodeval>$ = new ColumnDef($<stringval>1, dynamic_cast<SQLType*>($<nodeval>2), dynamic_cast<CompressDef*>($<nodeval>4), dynamic_cast<ColumnConstraintDef*>($<nodeval>3)); }
		| 
	;

opt_compression:
		 NAME NAME
		{
			if (!boost::iequals(*$<stringval>1, "encoding"))
				throw std::runtime_error("Invalid identifier " + *$<stringval>1 + " in column definition.");
			$<nodeval>$ = new CompressDef($<stringval>2, 0);
		}
		| NAME NAME '(' INTNUM ')'
		{
			if (!boost::iequals(*$<stringval>1, "encoding"))
				throw std::runtime_error("Invalid identifier " + *$<stringval>1 + " in column definition.");
			$<nodeval>$ = new CompressDef($<stringval>2, (int)$<intval>4);
		}
		| /* empty */ { $<nodeval>$ = nullptr; }
		;

column_constraint_def:
		NOT NULLX { $<nodeval>$ = new ColumnConstraintDef(true, false, false, nullptr); }
	|	NOT NULLX UNIQUE { $<nodeval>$ = new ColumnConstraintDef(true, true, false, nullptr); }
	|	NOT NULLX PRIMARY KEY { $<nodeval>$ = new ColumnConstraintDef(true, true, true, nullptr); }
	|	DEFAULT literal { $<nodeval>$ = new ColumnConstraintDef(false, false, false, dynamic_cast<Literal*>($<nodeval>2)); }
	|	DEFAULT NULLX { $<nodeval>$ = new ColumnConstraintDef(false, false, false, new NullLiteral()); }
	|	DEFAULT USER { $<nodeval>$ = new ColumnConstraintDef(false, false, false, new UserLiteral()); }
	|	CHECK '(' search_condition ')' { $<nodeval>$ = new ColumnConstraintDef(dynamic_cast<Expr*>($<nodeval>3)); }
	|	REFERENCES table { $<nodeval>$ = new ColumnConstraintDef($<stringval>2, nullptr); }
	|	REFERENCES table '(' column ')' { $<nodeval>$ = new ColumnConstraintDef($<stringval>2, $<stringval>4); }
	;

table_constraint_def:
		UNIQUE '(' column_commalist ')'
	{ $<nodeval>$ = new UniqueDef(false, $<slistval>3); }
	|	PRIMARY KEY '(' column_commalist ')'
	{ $<nodeval>$ = new UniqueDef(true, $<slistval>4); }
	|	FOREIGN KEY '(' column_commalist ')'
			REFERENCES table 
	{ $<nodeval>$ = new ForeignKeyDef($<slistval>4, $<stringval>7, nullptr); }
	|	FOREIGN KEY '(' column_commalist ')'
			REFERENCES table '(' column_commalist ')'
	{ $<nodeval>$ = new ForeignKeyDef($<slistval>4, $<stringval>7, $<slistval>9); }
	|	CHECK '(' search_condition ')'
	{ $<nodeval>$ = new CheckDef(dynamic_cast<Expr*>($<nodeval>3)); }
	;

column_commalist:
		column { $<slistval>$ = new std::list<std::string*>(1, $<stringval>1); }
	|	column_commalist ',' column
	{
		$<slistval>$ = $<slistval>1;
		$<slistval>$->push_back($<stringval>3);
	}
	;

opt_with_option_list:
		WITH '(' name_eq_value_list ')'
		{ $<listval>$ = $<listval>3; }
		| /* empty */
		{ $<listval>$ = nullptr; }
		;

create_view_statement:
		CREATE VIEW opt_if_not_exists table opt_column_commalist
		AS query_spec opt_with_check_option
		{
			$<nodeval>$ = new CreateViewStmt($<stringval>4, $<slistval>5, dynamic_cast<QuerySpec*>($<nodeval>7), $<boolval>8, false, nullptr, $<boolval>3);
		}
		| CREATE NAME VIEW opt_if_not_exists table opt_column_commalist opt_with_option_list
		AS query_spec
		{
			if (!boost::iequals(*$<stringval>2, "materialized"))
				throw std::runtime_error("Invalid word " + *$<stringval>2);
			$<nodeval>$ = new CreateViewStmt($<stringval>5, $<slistval>6, dynamic_cast<QuerySpec*>($<nodeval>9), false, true, reinterpret_cast<std::list<NameValueAssign*>*>($<listval>7), $<boolval>4);
		}
	;

refresh_view_statement:
		NAME NAME VIEW table
		{
			if (!boost::iequals(*$<stringval>1, "refresh"))
				throw std::runtime_error("Invalid word " + *$<stringval>1);
			if (!boost::iequals(*$<stringval>2, "materialized"))
				throw std::runtime_error("Invalid word " + *$<stringval>2);
			$<nodeval>$ = new RefreshViewStmt($<stringval>4);
		}
		;

drop_view_statement:
		DROP VIEW opt_if_exists table
		{
			$<nodeval>$ = new DropViewStmt($<stringval>4, $<boolval>3);
		}
		;
	
opt_with_check_option:
		/* empty */	{	$<boolval>$ = false; }
	|	WITH CHECK OPTION { $<boolval>$ = true; }
	;

opt_column_commalist:
		/* empty */ { $<slistval>$ = nullptr; }
	|	'(' column_commalist ')' { $<slistval>$ = $<slistval>2; }
	;

/* NOT SUPPORTED
privilege_def:
		GRANT privileges ON table TO grantee_commalist
		opt_with_grant_option
	;

opt_with_grant_option:
		// empty
	|	WITH GRANT OPTION
	;

privileges:
		ALL PRIVILEGES
	|	ALL
	|	operation_commalist
	;

operation_commalist:
		operation
	|	operation_commalist ',' operation
	;

operation:
		SELECT
	|	INSERT
	|	DELETE
	|	UPDATE opt_column_commalist
	|	REFERENCES opt_column_commalist
	;


grantee_commalist:
		grantee
	|	grantee_commalist ',' grantee
	;

grantee:
		PUBLIC
	|	user
	;

sql:
		cursor_def
	;


cursor_def:
		DECLARE cursor CURSOR FOR query_exp opt_order_by_clause
	;

NOT SUPPORTED */

opt_order_by_clause:
		/* empty */ { $<listval>$ = nullptr; }
	|	ORDER BY ordering_spec_commalist { $<listval>$ = $<listval>3; }
	;

ordering_spec_commalist:
		ordering_spec { $<listval>$ = new std::list<Node*>(1, $<nodeval>1); }
	|	ordering_spec_commalist ',' ordering_spec
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

ordering_spec:
		INTNUM opt_asc_desc opt_null_order
		{ $<nodeval>$ = new OrderSpec($<intval>1, nullptr, $<boolval>2, $<boolval>3); }
	|	column_ref opt_asc_desc opt_null_order
	{ $<nodeval>$ = new OrderSpec(0, dynamic_cast<ColumnRef*>($<nodeval>1), $<boolval>2, $<boolval>3); }
	;

opt_asc_desc:
		/* empty */ { $<boolval>$ = false; /* default is ASC */ }
	|	ASC { $<boolval>$ = false; }
	|	DESC { $<boolval>$ = true; }
	;

opt_null_order:
		/* empty */ { $<boolval>$ = false; /* default is NULL LAST */ }
	| NULLX FIRST { $<boolval>$ = true; }
	| NULLX LAST { $<boolval>$ = false; }
	;

	/* manipulative statements */

sql:		manipulative_statement
	;

manipulative_statement:
		/* close_statement */
	/* |	commit_statement */
	/* |	delete_statement_positioned */
		delete_statement
	/* |	fetch_statement */
	|	insert_statement
	/* |	open_statement */
	/* |	rollback_statement */
	|	select_statement
	/* |	update_statement_positioned */
	|	update_statement
	;

/* NOT SUPPORTED
close_statement:
		CLOSE cursor
	;

commit_statement:
		COMMIT WORK
	;

delete_statement_positioned:
		DELETE FROM table WHERE CURRENT OF cursor
	;

fetch_statement:
		FETCH cursor INTO target_commalist
	;
NOT SUPPORTED */

delete_statement:
		DELETE FROM table opt_where_clause
		{ $<nodeval>$ = new DeleteStmt($<stringval>3, dynamic_cast<Expr*>($<nodeval>4)); }
	;

insert_statement:
		INSERT INTO table opt_column_commalist VALUES '(' atom_commalist ')'
		{
			$<nodeval>$ = new InsertValuesStmt($<stringval>3, $<slistval>4, reinterpret_cast<std::list<Expr*>*>($<listval>7));
		}
		| INSERT INTO table opt_column_commalist query_spec
		{
			$<nodeval>$ = new InsertQueryStmt($<stringval>3, $<slistval>4, dynamic_cast<QuerySpec*>($<nodeval>5));
		}
	;

/* NOT SUPPORTED
open_statement:
		OPEN cursor
	;

rollback_statement:
		ROLLBACK WORK
	;

select_statement:
		SELECT opt_all_distinct selection
		INTO target_commalist
		table_exp
	;
NOT SUPPORTED */

opt_all_distinct:
		/* empty */ { $<boolval>$ = false; }
	|	ALL { $<boolval>$ = false; }
	|	DISTINCT { $<boolval>$ = true; }
	;

/* NOT SUPPORTED
update_statement_positioned:
		UPDATE table SET assignment_commalist
		WHERE CURRENT OF cursor
	;
NOT SUPPORTED */

assignment_commalist:
		assignment
	{ $<listval>$ = new std::list<Node*>(1, $<nodeval>1); }
	|	assignment_commalist ',' assignment
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

assignment:
		column EQUAL scalar_exp 
		{ $<nodeval>$ = new Assignment($<stringval>1, dynamic_cast<Expr*>($<nodeval>3)); }
	;

update_statement:
		UPDATE table SET assignment_commalist opt_where_clause
		{ $<nodeval>$ = new UpdateStmt($<stringval>2, reinterpret_cast<std::list<Assignment*>*>($<listval>4), dynamic_cast<Expr*>($<nodeval>5)); }
	;

/* NOT SUPPORTED
target_commalist:
		target
	|	target_commalist ',' target
	;

target:
		parameter_ref
	;
NOT SUPPORTED */

opt_where_clause:
		/* empty */ { $<nodeval>$ = nullptr; }
	|	where_clause { $<nodeval>$ = $<nodeval>1; }
	;

opt_limit_clause:
		LIMIT INTNUM { $<intval>$ = $<intval>2; }
	| LIMIT ALL { $<intval>$ = 0; /* 0 means ALL */ }
	| /* empty */ { $<intval>$ = 0; /* 0 means ALL */ }
	;
opt_offset_clause:
		OFFSET INTNUM { $<intval>$ = $<intval>2; }
	| OFFSET INTNUM NAME
	{
		if (!boost::iequals(*$<stringval>3, "row") && !boost::iequals(*$<stringval>3, "rows"))
			throw std::runtime_error("Invalid word in OFFSET clause " + *$<stringval>3);
		$<intval>$ = $<intval>2;
	}
	| /* empty */ 
	{
		$<intval>$ = 0;
	}
	;

select_statement:
		query_exp opt_order_by_clause opt_limit_clause opt_offset_clause
		{ $<nodeval>$ = new SelectStmt(dynamic_cast<QueryExpr*>($<nodeval>1), reinterpret_cast<std::list<OrderSpec*>*>($<listval>2), $<intval>3, $<intval>4); }
	;

	/* query expressions */

query_exp:
		query_term { $<nodeval>$ = $<nodeval>1; }
	|	query_exp UNION query_term 
	{ $<nodeval>$ = new UnionQuery(false, dynamic_cast<QueryExpr*>($<nodeval>1), dynamic_cast<QueryExpr*>($<nodeval>3)); }
	|	query_exp UNION ALL query_term
	{ $<nodeval>$ = new UnionQuery(true, dynamic_cast<QueryExpr*>($<nodeval>1), dynamic_cast<QueryExpr*>($<nodeval>4)); }
	;

query_term:
		query_spec { $<nodeval>$ = $<nodeval>1; }
	|	'(' query_exp ')' { $<nodeval>$ = $<nodeval>2; }
	;

query_spec:
		SELECT opt_all_distinct selection from_clause opt_where_clause opt_group_by_clause opt_having_clause
		{ $<nodeval>$ = new QuerySpec($<boolval>2,
																	reinterpret_cast<std::list<SelectEntry*>*>($<listval>3),
																	reinterpret_cast<std::list<TableRef*>*>($<listval>4),
																	dynamic_cast<Expr*>($<nodeval>5),
																	reinterpret_cast<std::list<Expr*>*>($<listval>6),
																	dynamic_cast<Expr*>($<nodeval>7));
		}
	;

selection:
		select_entry_commalist { $<listval>$ = $<listval>1; }
	|	'*' { $<listval>$ = nullptr; /* nullptr means SELECT * */ }
	;

from_clause:
		FROM table_ref_commalist { $<listval>$ = $<listval>2; }
	;

table_ref_commalist:
		table_ref { $<listval>$ = new std::list<Node*>(1, $<nodeval>1); }
	|	table_ref_commalist ',' table_ref
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

table_ref:
		table { $<nodeval>$ = new TableRef($<stringval>1); }
	|	table range_variable { $<nodeval>$ = new TableRef($<stringval>1, $<stringval>2); }
	;

where_clause:
		WHERE search_condition { $<nodeval>$ = $<nodeval>2; }
	;

opt_group_by_clause:
		/* empty */ { $<listval>$ = nullptr; }
	|	GROUP BY exp_commalist { $<listval>$ = $<listval>3; }
	;

exp_commalist:
		scalar_exp { $<listval>$ = new std::list<Node*>(1, $<nodeval>1); }
	|	exp_commalist ',' scalar_exp
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

opt_having_clause:
		/* empty */ { $<nodeval>$ = nullptr; }
	|	HAVING search_condition { $<nodeval>$ = $<nodeval>2; }
	;

	/* search conditions */

search_condition:
	|	search_condition OR search_condition
	{ $<nodeval>$ = new OperExpr(kOR, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	search_condition AND search_condition
	{ $<nodeval>$ = new OperExpr(kAND, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	NOT search_condition
	{ $<nodeval>$ = new OperExpr(kNOT, dynamic_cast<Expr*>($<nodeval>2), nullptr); }
	|	'(' search_condition ')' { $<nodeval>$ = $<nodeval>2; }
	|	predicate { $<nodeval>$ = $<nodeval>1; }
	;

predicate:
		comparison_predicate { $<nodeval>$ = $<nodeval>1; }
	|	between_predicate { $<nodeval>$ = $<nodeval>1; }
	|	like_predicate { $<nodeval>$ = $<nodeval>1; }
	|	test_for_null { $<nodeval>$ = $<nodeval>1; }
	|	in_predicate { $<nodeval>$ = $<nodeval>1; }
	|	all_or_any_predicate { $<nodeval>$ = $<nodeval>1; }
	|	existence_test { $<nodeval>$ = $<nodeval>1; }
	;

comparison_predicate:
		scalar_exp comparison scalar_exp
		{ $<nodeval>$ = new OperExpr($<opval>2, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	scalar_exp comparison subquery
		{ 
			$<nodeval>$ = new OperExpr($<opval>2, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); 
			/* subquery can only return a single result */
			dynamic_cast<SubqueryExpr*>($<nodeval>3)->set_qualifier(kONE); 
		}
	;

between_predicate:
		scalar_exp NOT BETWEEN scalar_exp AND scalar_exp
		{ $<nodeval>$ = new BetweenExpr(true, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>4), dynamic_cast<Expr*>($<nodeval>6)); }
	|	scalar_exp BETWEEN scalar_exp AND scalar_exp
		{ $<nodeval>$ = new BetweenExpr(false, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3), dynamic_cast<Expr*>($<nodeval>5)); }
	;

like_predicate:
		scalar_exp NOT LIKE atom opt_escape
	{ $<nodeval>$ = new LikeExpr(true, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>4), dynamic_cast<Expr*>($<nodeval>5)); }
	|	scalar_exp LIKE atom opt_escape
	{ $<nodeval>$ = new LikeExpr(false, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3), dynamic_cast<Expr*>($<nodeval>4)); }
	;

opt_escape:
		/* empty */ { $<nodeval>$ = nullptr; }
	|	ESCAPE atom { $<nodeval>$ = $<nodeval>2; }
	;

test_for_null:
		column_ref IS NOT NULLX { $<nodeval>$ = new IsNullExpr(true, dynamic_cast<Expr*>($<nodeval>1)); }
	|	column_ref IS NULLX { $<nodeval>$ = new IsNullExpr(false, dynamic_cast<Expr*>($<nodeval>1)); }
	;

in_predicate:
		scalar_exp NOT IN subquery
		{ $<nodeval>$ = new InSubquery(true, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<SubqueryExpr*>($<nodeval>4)); }
	|	scalar_exp IN subquery
		{ $<nodeval>$ = new InSubquery(false, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<SubqueryExpr*>($<nodeval>3)); }
	|	scalar_exp NOT IN '(' atom_commalist ')'
	{ $<nodeval>$ = new InValues(true, dynamic_cast<Expr*>($<nodeval>1), reinterpret_cast<std::list<Expr*>*>($<listval>5)); }
	|	scalar_exp IN '(' atom_commalist ')'
	{ $<nodeval>$ = new InValues(false, dynamic_cast<Expr*>($<nodeval>1), reinterpret_cast<std::list<Expr*>*>($<listval>4)); }
	;

atom_commalist:
		atom { $<listval>$ = new std::list<Node*>(1, $<nodeval>1); }
	|	atom_commalist ',' atom
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

all_or_any_predicate:
		scalar_exp comparison any_all_some subquery
		{
			$<nodeval>$ = new OperExpr($<opval>2, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>4));
			dynamic_cast<SubqueryExpr*>($<nodeval>4)->set_qualifier($<qualval>3);
		}
	;

comparison:
	EQUAL { $<opval>$ = $<opval>1; }
	| COMPARISON { $<opval>$ = $<opval>1; }
	;
			
any_all_some:
		ANY
	|	ALL
	|	SOME
	;

existence_test:
		EXISTS subquery { $<nodeval>$ = new ExistsExpr(dynamic_cast<QuerySpec*>($<nodeval>2)); }
	;

subquery:
		'(' query_spec ')' { $<nodeval>$ = new SubqueryExpr(dynamic_cast<QuerySpec*>($<nodeval>2)); }
	;

when_then_list:
		WHEN search_condition THEN scalar_exp
		{
			$<listval>$ = new std::list<Node*>(1, new ExprPair(dynamic_cast<Expr*>($<nodeval>2), dynamic_cast<Expr*>($<nodeval>4)));
		}
		| when_then_list WHEN search_condition THEN scalar_exp
		{
			$<listval>$ = $<listval>1;
			$<listval>$->push_back(new ExprPair(dynamic_cast<Expr*>($<nodeval>3), dynamic_cast<Expr*>($<nodeval>5)));
		}
		;
opt_else_expr :
		ELSE scalar_exp { $<nodeval>$ = $<nodeval>2; }
		| /* empty */ { $<nodeval>$ = nullptr; }
		;

case_exp: CASE when_then_list opt_else_expr END
	{
		$<nodeval>$ = new CaseExpr(reinterpret_cast<std::list<ExprPair*>*>($<listval>2), dynamic_cast<Expr*>($<nodeval>3));
	}
	;

	/* scalar expressions */

scalar_exp:
		scalar_exp '+' scalar_exp { $<nodeval>$ = new OperExpr(kPLUS, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	scalar_exp '-' scalar_exp { $<nodeval>$ = new OperExpr(kMINUS, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	scalar_exp '*' scalar_exp { $<nodeval>$ = new OperExpr(kMULTIPLY, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	scalar_exp '/' scalar_exp { $<nodeval>$ = new OperExpr(kDIVIDE, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	'+' scalar_exp %prec UMINUS { $<nodeval>$ = $<nodeval>2; }
	|	'-' scalar_exp %prec UMINUS { $<nodeval>$ = new OperExpr(kUMINUS, dynamic_cast<Expr*>($<nodeval>2), nullptr); }
	|	atom { $<nodeval>$ = $<nodeval>1; }
	|	column_ref { $<nodeval>$ = $<nodeval>1; }
	|	function_ref { $<nodeval>$ = $<nodeval>1; }
	|	'(' scalar_exp ')' { $<nodeval>$ = $<nodeval>2; }
	| CAST '(' scalar_exp AS data_type ')'
	{ $<nodeval>$ = new CastExpr(dynamic_cast<Expr*>($<nodeval>3), dynamic_cast<SQLType*>($<nodeval>5)); }
	| case_exp { $<nodeval>$ = $<nodeval>1; }
	;

select_entry: scalar_exp { $<nodeval>$ = new SelectEntry(dynamic_cast<Expr*>($<nodeval>1), nullptr); }
	| scalar_exp NAME { $<nodeval>$ = new SelectEntry(dynamic_cast<Expr*>($<nodeval>1), $<stringval>2); }
	| scalar_exp AS NAME { $<nodeval>$ = new SelectEntry(dynamic_cast<Expr*>($<nodeval>1), $<stringval>3); }
	;

select_entry_commalist:
		select_entry { $<listval>$ = new std::list<Node*>(1, $<nodeval>1); }
	|	select_entry_commalist ',' select_entry
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

atom:
		literal { $<nodeval>$ = $<nodeval>1; }
	|	USER { $<nodeval>$ = new UserLiteral(); }
	| NULLX { $<nodeval>$ = new NullLiteral(); }
	/* |	parameter_ref { $<nodeval>$ = $<nodeval>1; } */
	;

/* TODO: do Postgres style PARAM
parameter_ref:
		parameter
	|	parameter parameter
	|	parameter INDICATOR parameter
	;
*/

function_ref:
		NAME '(' '*' ')' { $<nodeval>$ = new FunctionRef($<stringval>1); }
	|	NAME '(' DISTINCT column_ref ')' { $<nodeval>$ = new FunctionRef($<stringval>1, true, dynamic_cast<Expr*>($<nodeval>4)); }
	|	NAME '(' ALL scalar_exp ')' { $<nodeval>$ = new FunctionRef($<stringval>1, dynamic_cast<Expr*>($<nodeval>4)); }
	|	NAME '(' scalar_exp ')' { $<nodeval>$ = new FunctionRef($<stringval>1, dynamic_cast<Expr*>($<nodeval>3)); }
	;

literal:
		STRING { $<nodeval>$ = new StringLiteral($<stringval>1); }
	|	INTNUM { $<nodeval>$ = new IntLiteral($<intval>1); }
	|	FIXEDNUM { $<nodeval>$ = new FixedPtLiteral($<stringval>1); }
	| FLOAT { $<nodeval>$ = new FloatLiteral($<floatval>1); }
	| DOUBLE { $<nodeval>$ = new DoubleLiteral($<doubleval>1); }
	| data_type STRING
	{ $<nodeval>$ = new CastExpr(new StringLiteral($<stringval>2), dynamic_cast<SQLType*>($<nodeval>1)); }
	;

	/* miscellaneous */

table:
		NAME { $<stringval>$ = $<stringval>1; }
	/* |	NAME '.' NAME { $$ = new TableRef($<stringval>1, $<stringval>3); } */
	;

column_ref:
		NAME { $<nodeval>$ = new ColumnRef($<stringval>1); }
	|	NAME '.' NAME	{ $<nodeval>$ = new ColumnRef($<stringval>1, $<stringval>3); }
	| NAME '.' '*' { $<nodeval>$ = new ColumnRef($<stringval>1, nullptr); }
	/* |	NAME '.' NAME '.' NAME { $$ = new ColumnRef($<stringval>1, $<stringval>3, $<stringval>5); } */
	;

non_neg_int: INTNUM 
		{ 
			if ($<intval>1 < 0)
				throw std::runtime_error("No negative number in type definition.");
			$<intval>$ = $<intval>1; 
		}

		/* data types */

data_type:
		BIGINT { $<nodeval>$ = new SQLType(kBIGINT); }
	| TEXT { $<nodeval>$ = new SQLType(kTEXT); }
	|	BOOLEAN { $<nodeval>$ = new SQLType(kBOOLEAN); }
	|	CHARACTER { $<nodeval>$ = new SQLType(kCHAR); }
	|	CHARACTER '(' non_neg_int ')' { $<nodeval>$ = new SQLType(kCHAR, $<intval>3); }
	|	NUMERIC { $<nodeval>$ = new SQLType(kNUMERIC); }
	|	NUMERIC '(' non_neg_int ')' { $<nodeval>$ = new SQLType(kNUMERIC, $<intval>3); }
	|	NUMERIC '(' non_neg_int ',' non_neg_int ')' { $<nodeval>$ = new SQLType(kNUMERIC, $<intval>3, $<intval>5); }
	|	DECIMAL { $<nodeval>$ = new SQLType(kDECIMAL); }
	|	DECIMAL '(' non_neg_int ')' { $<nodeval>$ = new SQLType(kDECIMAL, $<intval>3); }
	|	DECIMAL '(' non_neg_int ',' non_neg_int ')' { $<nodeval>$ = new SQLType(kDECIMAL, $<intval>3, $<intval>5); }
	|	INTEGER { $<nodeval>$ = new SQLType(kINT); }
	|	SMALLINT { $<nodeval>$ = new SQLType(kSMALLINT); }
	|	FLOAT { $<nodeval>$ = new SQLType(kFLOAT); }
	/* |	FLOAT '(' non_neg_int ')' { $<nodeval>$ = new SQLType(kFLOAT, $<intval>3); } */
	|	REAL { $<nodeval>$ = new SQLType(kFLOAT); }
	|	DOUBLE PRECISION { $<nodeval>$ = new SQLType(kDOUBLE); }
	|	DOUBLE { $<nodeval>$ = new SQLType(kDOUBLE); }
	| DATE { $<nodeval>$ = new SQLType(kDATE); }
	| TIME { $<nodeval>$ = new SQLType(kTIME); }
	| TIME '(' non_neg_int ')' { $<nodeval>$ = new SQLType(kTIME, $<intval>3); }
	| TIMESTAMP { $<nodeval>$ = new SQLType(kTIMESTAMP); }
	| TIMESTAMP '(' non_neg_int ')' { $<nodeval>$ = new SQLType(kTIMESTAMP, $<intval>3); }
	;

	/* the various things you can name */

column:		NAME { $<stringval>$ = $<stringval>1; }
	;

/*
cursor:		NAME { $<stringval>$ = $<stringval>1; }
	;
*/

/* TODO: do Postgres-styl PARAM
parameter:
		PARAMETER	// :name handled in parser
		{ $$ = new Parameter(yytext+1); }
	;
*/

range_variable:	NAME { $<stringval>$ = $<stringval>1; }
	;

/* 
user:		NAME { $<stringval>$ = $<stringval>1; }
	;
*/

%%

/**
 * @file	Visitor.h
 * @author	Steven Stewart <steve@map-d.com>
 *
 * This header file specifies the void Visitor API for the SQL parser.
 */
#ifndef SQL_VISITOR_H
#define SQL_VISITOR_H


namespace SQL_Namespace {
    
    // forward declarations
    class AggrExpr;
    class AlterStmt;
    class Column;
    class ColumnDef;
    class ColumnDefList;
    class ColumnList;
    class Comparison;
    class CreateStmt;
    class DeleteStmt;
    class DdlStmt;
    class DmlStmt;
    class DropStmt;
    class FromClause;
    class InsertColumnList;
    class InsertStmt;
    class Literal;
    class LiteralList;
    class MapdDataT;
    class MathExpr;
    class OptAllDistinct;
    class OptGroupby;
    class OptHaving;
    class OptLimit;
    class OptOrderby;
    class OptWhere;
    class OrderbyColumn;
    class OrderbyColumnList;
    class Predicate;
    class RenameStmt;
    class ScalarExpr;
    class ScalarExprList;
    class SearchCondition;
    class SelectStmt;
    class Selection;
    class SqlStmt;
    class Table;
    class TableList;
    class sqlStmt;
    
    /**
     * @class Visitor
     * @brief This is the Visitor class.
     */
    class Visitor {
        
    public:
        
        virtual void visit(AggrExpr*) {}
        virtual void visit(AlterStmt*) {}
        virtual void visit(Column*) {}
        virtual void visit(ColumnDef*) {}
        virtual void visit(ColumnDefList*) {}
        virtual void visit(ColumnList*) {}
        virtual void visit(Comparison*) {}
        virtual void visit(CreateStmt*) {}
        virtual void visit(DeleteStmt*) {}
        virtual void visit(DdlStmt*) {}
        virtual void visit(DmlStmt*) {}
        virtual void visit(DropStmt*) {}
        virtual void visit(FromClause*) {}
        virtual void visit(InsertColumnList*) {}
        virtual void visit(InsertStmt*) {}
        virtual void visit(Literal*) {}
        virtual void visit(LiteralList*) {}
        virtual void visit(MapdDataT*) {}
        virtual void visit(MathExpr*) {}
        virtual void visit(OptAllDistinct*) {}
        virtual void visit(OptGroupby*) {}
        virtual void visit(OptHaving*) {}
        virtual void visit(OptLimit*) {}
        virtual void visit(OptOrderby*) {}
        virtual void visit(OptWhere*) {}
        virtual void visit(OrderbyColumn*) {}
        virtual void visit(OrderbyColumnList*) {}
        virtual void visit(Predicate*) {}
        virtual void visit(RenameStmt*) {}
        virtual void visit(ScalarExpr*) {}
        virtual void visit(ScalarExprList*) {}
        virtual void visit(SearchCondition*) {}
        virtual void visit(Selection*) {}
        virtual void visit(SelectStmt*) {}
        virtual void visit(SqlStmt*) {}
        virtual void visit(Table*) {}
        virtual void visit(TableList*) {}
        
    };
    
} // SQL_Namespace

#endif // SQL_VISITOR_H

/*
 * Some cool MapD Header
 */
package com.mapd.calcite.parser;

import org.apache.calcite.avatica.util.Casing;
import org.apache.calcite.avatica.util.Quoting;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.sql.SqlAsOperator;
import org.apache.calcite.sql.SqlBasicCall;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlOrderBy;
import org.apache.calcite.sql.SqlSelect;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlTypeFactoryImpl;
import org.apache.calcite.sql.validate.SqlConformance;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql2rel.SqlToRelConverter;
import org.apache.calcite.sql2rel.StandardConvertletTable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */


public final class MapDParser {

  final static Logger MAPDLOGGER = LoggerFactory.getLogger(MapDParser.class);

  private final Quoting quoting = Quoting.DOUBLE_QUOTE;
  private final Casing unquotedCasing = Casing.UNCHANGED;
  private final Casing quotedCasing = Casing.UNCHANGED;

  private final RelDataTypeFactory typeFactory;
  private final MapDCatalogReader catalogReader;
  private final SqlValidator validator;
  private final SqlToRelConverter converter;

  public MapDParser(){
    typeFactory = new SqlTypeFactoryImpl(RelDataTypeSystem.DEFAULT);
    catalogReader = new MapDCatalogReader(typeFactory);
    validator = new MapDValidator(
                  createOperatorTable(),
                  catalogReader,
                  typeFactory,
                  SqlConformance.DEFAULT);
    final RexBuilder rexBuilder = new RexBuilder(typeFactory);
    final RelOptCluster cluster = RelOptCluster.create(new MapDRelOptPlanner(), rexBuilder);
    converter = new SqlToRelConverter(null, validator, catalogReader, cluster,
                  StandardConvertletTable.INSTANCE);
  }

  public String getRelAlgebra(String sql, final boolean legacy_syntax, final MapDUser mapDUser)
          throws SqlParseException {
    SqlNode node = processSQL(sql, legacy_syntax);

    boolean is_select_star = isSelectStar(node);
    catalogReader.setCurrentMapDUser(mapDUser);
    SqlNode validate = validator.validate(node);

    SqlSelect validate_select = getSelectChild(validate);
    // Hide rowid from select * queries
    if (legacy_syntax && is_select_star && validate_select != null) {
      SqlNodeList proj_exprs = ((SqlSelect) validate).getSelectList();
      SqlNodeList new_proj_exprs = new SqlNodeList(proj_exprs.getParserPosition());
      for (SqlNode proj_expr : proj_exprs) {
        if (proj_expr instanceof SqlIdentifier
                && (((SqlIdentifier) proj_expr).toString().toLowerCase()).endsWith(".rowid")) {
          continue;
        }
        new_proj_exprs.add(proj_expr);
      }
      validate_select.setSelectList(new_proj_exprs);
    }

    final RelRoot sqlRel = converter.convertQuery(node, true, true);
    //final RelNode sqlRel = converter.convertSelect((SqlSelect)node, true);
    //RexNode convertExpression = converter.convertExpression(node);

    //logger.debug("After convert relNode is "+ convertExpression.toString());
    //logger.debug("After convert relRoot kind is " + sqlRel.kind);
    //logger.debug("After convert relRoot project is " + sqlRel.project().toString());
    //logger.debug("After convert relalgebra is \n" + RelOptUtil.toString(sqlRel.project()));
    RelNode project = sqlRel.project();

    String res = MapDSerializer.toString(project);

    //logger.info("After convert relalgebra is \n" + res);
    return res;
  }

  private boolean isSelectStar(SqlNode node) {
    SqlSelect select_node = getSelectChild(node);
    if (select_node == null) {
      return false;
    }
    SqlNodeList proj_exprs = select_node.getSelectList();
    if (proj_exprs.size() != 1) {
      return false;
    }
    SqlNode proj_expr = proj_exprs.get(0);
    if (!(proj_expr instanceof SqlIdentifier)) {
      return false;
    }
    return ((SqlIdentifier) proj_expr).isStar();
  }

  private SqlSelect getSelectChild(SqlNode node) {
    if (node instanceof SqlSelect) {
      return (SqlSelect) node;
    }
    if (node instanceof SqlOrderBy) {
      SqlOrderBy order_by_node = (SqlOrderBy) node;
      if (order_by_node.query instanceof SqlSelect) {
        return (SqlSelect) order_by_node.query;
      }
    }
    return null;
  }

  private SqlNode processSQL(String sql, final boolean legacy_syntax) throws SqlParseException {
    SqlNode node = null;
    SqlParser sqlp = getSqlParser(sql);
    try {
      node = sqlp.parseStmt();
      MAPDLOGGER.debug(" node is \n" + node.toString());
    } catch (SqlParseException ex) {
      MAPDLOGGER.error("failed to process SQL '" + sql + "' \n" + ex.toString());
      throw ex;
    }
    if (!legacy_syntax) {
      return node;
    }
    SqlSelect select_node = null;
    if (node instanceof SqlSelect) {
      select_node = (SqlSelect) node;
    } else if (node instanceof SqlOrderBy) {
      SqlOrderBy order_by_node = (SqlOrderBy) node;
      if (order_by_node.query instanceof SqlSelect) {
        select_node = (SqlSelect) order_by_node.query;
      }
    }
    if (select_node != null) {
      desugar(select_node);
    }
    return node;
  }

  private static void desugar(SqlSelect select_node) {
    SqlNodeList select_list = select_node.getSelectList();
    java.util.Map<String, SqlNode> id_to_expr = new java.util.HashMap<String, SqlNode>();
    for (SqlNode proj : select_list) {
      if (!(proj instanceof SqlBasicCall)) {
        continue;
      }
      SqlBasicCall proj_call = (SqlBasicCall) proj;
      if (proj_call.getOperator() instanceof SqlAsOperator) {
        SqlNode[] operands = proj_call.getOperands();
        SqlIdentifier id = (SqlIdentifier) operands[1];
        id_to_expr.put(id.toString(), operands[0]);
      }
    }
    SqlNodeList group_by_list = select_node.getGroup();
    if (group_by_list == null) {
      return;
    }
    select_node.setGroupBy(expandAliases(group_by_list, id_to_expr));
    SqlNode having = select_node.getHaving();
    if (having == null) {
      return;
    }
    expandAliases(having, id_to_expr);
  }

  private static SqlNode expandAliases(final SqlNode node,
          final java.util.Map<String, SqlNode> id_to_expr) {
    if (node instanceof SqlIdentifier && id_to_expr.containsKey(node.toString())) {
      return id_to_expr.get(node.toString());
    }
    if (node instanceof SqlBasicCall) {
      SqlBasicCall node_call = (SqlBasicCall) node;
      SqlNode[] operands = node_call.getOperands();
      for (int i = 0; i < operands.length; ++i) {
        node_call.setOperand(i, expandAliases(operands[i], id_to_expr));
      }
    }
    return node;
  }

  private static SqlNodeList expandAliases(final SqlNodeList group_by_list, final java.util.Map<String, SqlNode> id_to_expr) {
    SqlNodeList new_group_by_list = new SqlNodeList(new SqlParserPos(-1, -1));
    for (SqlNode group_by : group_by_list) {
      if (!(group_by instanceof SqlIdentifier)) {
        new_group_by_list.add(group_by);
        continue;
      }
      SqlIdentifier group_by_id = ((SqlIdentifier) group_by);
      if (id_to_expr.containsKey(group_by_id.toString())) {
        new_group_by_list.add(id_to_expr.get(group_by_id.toString()));
      } else {
        new_group_by_list.add(group_by);
      }
    }
    return new_group_by_list;
  }

  /**
   * Creates an operator table.
   *
   * @return New operator table
   */
  protected SqlOperatorTable createOperatorTable() {
    final MapDSqlOperatorTable tempOpTab
            = new MapDSqlOperatorTable(SqlStdOperatorTable.instance());
    // MAT 11 Nov 2015
    // Example of how to add custom function
    MapDSqlOperatorTable.addUDF(tempOpTab);
    return tempOpTab;
  }

  protected SqlNode parseStmt(String sql) throws SqlParseException {
    return getSqlParser(sql).parseStmt();
  }

  protected SqlParser getSqlParser(String sql) {
    return SqlParser.create(sql,
            SqlParser.configBuilder()
            .setQuoting(quoting)
            .setUnquotedCasing(unquotedCasing)
            .setQuotedCasing(quotedCasing)
            .build());
  }
}
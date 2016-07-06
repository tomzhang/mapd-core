/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mapd.calcite.parser;

import java.util.Map;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.mapd.parser.server.ExtensionFunction;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.util.ChainedSqlOperatorTable;
import org.apache.calcite.sql.util.ListSqlOperatorTable;

/**
 *
 * @author michael
 */
public class MapDSqlOperatorTable extends ChainedSqlOperatorTable {

    /**
     * Mock operator table for testing purposes. Contains the standard SQL
     * operator table, plus a list of operators.
     */
    //~ Instance fields --------------------------------------------------------
    private final ListSqlOperatorTable listOpTab;

    //~ Constructors -----------------------------------------------------------
    public MapDSqlOperatorTable(SqlOperatorTable parentTable) {
        super(ImmutableList.of(parentTable, new ListSqlOperatorTable()));
        listOpTab = (ListSqlOperatorTable) tableList.get(1);
    }

    //~ Methods ----------------------------------------------------------------
    /**
     * Adds an operator to this table.
     *
     * @param op
     */
    public void addOperator(SqlOperator op) {
        listOpTab.add(op);
    }

    // MAT Nov 11 2015
    // These are example of how to add custom functions
    // left in as a starting point for when we need them
    public static void addUDF(MapDSqlOperatorTable opTab, final Map<String, ExtensionFunction> extSigs) {
        // Don't use anonymous inner classes. They can't be instantiated
        // using reflection when we are deserializing from JSON.
        //opTab.addOperator(new RampFunction());
        //opTab.addOperator(new DedupFunction());
        opTab.addOperator(new MyUDFFunction());
        opTab.addOperator(new PgUnnest());
        opTab.addOperator(new Any());
        opTab.addOperator(new All());
        opTab.addOperator(new Now());
        opTab.addOperator(new Datetime());
        opTab.addOperator(new PgExtract());
        opTab.addOperator(new PgDateTrunc());
        opTab.addOperator(new Length());
        opTab.addOperator(new CharLength());
        opTab.addOperator(new PgILike());
        if (extSigs == null) {
          return;
        }
        for (Map.Entry<String, ExtensionFunction> extSig : extSigs.entrySet()) {
          opTab.addOperator(new ExtFunction(extSig.getKey(), extSig.getValue()));
        }
    }

    /**
     * "RAMP" user-defined function.
     */
    public static class RampFunction extends SqlFunction {

        public RampFunction() {
            super("RAMP",
                    SqlKind.OTHER_FUNCTION,
                    null,
                    null,
                    OperandTypes.NUMERIC,
                    SqlFunctionCategory.USER_DEFINED_FUNCTION);
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.builder()
                    .add("I", SqlTypeName.INTEGER)
                    .build();
        }
    }

    /**
     * "DEDUP" user-defined function.
     */
    public static class DedupFunction extends SqlFunction {

        public DedupFunction() {
            super("DEDUP",
                    SqlKind.OTHER_FUNCTION,
                    null,
                    null,
                    OperandTypes.VARIADIC,
                    SqlFunctionCategory.USER_DEFINED_FUNCTION);
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.builder()
                    .add("NAME", SqlTypeName.VARCHAR, 1024)
                    .build();
        }
    }

    /**
     * "MyUDFFunction" user-defined function test. our udf's will look like
     * system functions to calcite as it has no access to the code
     */
    public static class MyUDFFunction extends SqlFunction {

        public MyUDFFunction() {
            super("MyUDF",
                    SqlKind.OTHER_FUNCTION,
                    null,
                    null,
                    OperandTypes.STRING_STRING,
                    SqlFunctionCategory.SYSTEM);
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.createSqlType(SqlTypeName.BIGINT);
        }
    }

    /* Postgres-style UNNEST */
    public static class PgUnnest extends SqlFunction {

        public PgUnnest() {
            super("PG_UNNEST",
                    SqlKind.OTHER_FUNCTION,
                    null,
                    null,
                    OperandTypes.ARRAY,
                    SqlFunctionCategory.SYSTEM);
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            assert opBinding.getOperandCount() == 1;
            RelDataType elem_type = opBinding.getOperandType(0).getComponentType();
            assert elem_type != null;
            return elem_type;
        }
    }

    /* ANY qualifier */
    public static class Any extends SqlFunction {

        public Any() {
            super("PG_ANY",
                    SqlKind.OTHER_FUNCTION,
                    null,
                    null,
                    OperandTypes.ARRAY,
                    SqlFunctionCategory.SYSTEM);
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            assert opBinding.getOperandCount() == 1;
            RelDataType elem_type = opBinding.getOperandType(0).getComponentType();
            assert elem_type != null;
            return elem_type;
        }
    }

    /* ALL qualifier */
    public static class All extends SqlFunction {

        public All() {
            super("PG_ALL",
                    SqlKind.OTHER_FUNCTION,
                    null,
                    null,
                    OperandTypes.ARRAY,
                    SqlFunctionCategory.SYSTEM);
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            assert opBinding.getOperandCount() == 1;
            RelDataType elem_type = opBinding.getOperandType(0).getComponentType();
            assert elem_type != null;
            return elem_type;
        }
    }

    /* NOW() */
    public static class Now extends SqlFunction {

        public Now() {
            super("NOW",
                    SqlKind.OTHER_FUNCTION,
                    null,
                    null,
                    OperandTypes.NILADIC,
                    SqlFunctionCategory.SYSTEM);
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            assert opBinding.getOperandCount() == 0;
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.createSqlType(SqlTypeName.TIMESTAMP);
        }
    }

    /* DATETIME */
    public static class Datetime extends SqlFunction {

        public Datetime() {
            super("DATETIME",
                    SqlKind.OTHER_FUNCTION,
                    null,
                    null,
                    OperandTypes.STRING,
                    SqlFunctionCategory.SYSTEM);
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            assert opBinding.getOperandCount() == 1;
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.createSqlType(SqlTypeName.TIMESTAMP);
        }
    }

    /* Postgres-style EXTRACT */
    public static class PgExtract extends SqlFunction {

        public PgExtract() {
            super("PG_EXTRACT",
                    SqlKind.OTHER_FUNCTION,
                    null,
                    null,
                    OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.DATETIME),
                    SqlFunctionCategory.SYSTEM);
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.createSqlType(SqlTypeName.BIGINT);
        }
    }

    /* Postgres-style DATE_TRUNC */
    public static class PgDateTrunc extends SqlFunction {

        public PgDateTrunc() {
            super("PG_DATE_TRUNC",
                    SqlKind.OTHER_FUNCTION,
                    null,
                    null,
                    OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.DATETIME),
                    SqlFunctionCategory.SYSTEM);
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.createSqlType(SqlTypeName.TIMESTAMP);
        }
    }

    public static class Length extends SqlFunction {

        public Length() {
            super("LENGTH",
                    SqlKind.OTHER_FUNCTION,
                    null,
                    null,
                    OperandTypes.STRING,
                    SqlFunctionCategory.SYSTEM);
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.createSqlType(SqlTypeName.INTEGER);
        }
    }

    public static class CharLength extends SqlFunction {

        public CharLength() {
            super("CHAR_LENGTH",
                    SqlKind.OTHER_FUNCTION,
                    null,
                    null,
                    OperandTypes.STRING,
                    SqlFunctionCategory.SYSTEM);
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.createSqlType(SqlTypeName.INTEGER);
        }
    }

    public static class PgILike extends SqlFunction {

        public PgILike() {
            super("PG_ILIKE",
                    SqlKind.OTHER_FUNCTION,
                    null,
                    null,
                    OperandTypes.family(getSignatureFamilies(), new EscapeOptional()),
                    SqlFunctionCategory.SYSTEM);
        }

        private static java.util.List<SqlTypeFamily> getSignatureFamilies() {
            java.util.ArrayList<SqlTypeFamily> families = new java.util.ArrayList<SqlTypeFamily>();
            families.add(SqlTypeFamily.STRING);
            families.add(SqlTypeFamily.STRING);
            families.add(SqlTypeFamily.STRING);
            return families;
        }

        private static class EscapeOptional implements Predicate<Integer> {

            @Override
            public boolean apply(Integer t) {
                return t == 2;
            }
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.createSqlType(SqlTypeName.BOOLEAN);
        }
    }

    static class ExtFunction extends SqlFunction {
      ExtFunction(final String name, final ExtensionFunction sig) {
        super(name,
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(toSqlSignature(sig)),
              SqlFunctionCategory.SYSTEM);
        ret = toSqlTypeName(sig.getRet());
      }

      private static java.util.List<SqlTypeFamily> toSqlSignature(final ExtensionFunction sig) {
        java.util.List<SqlTypeFamily> sql_sig = new java.util.ArrayList<SqlTypeFamily>();
        for (final ExtensionFunction.ExtArgumentType arg_type : sig.getArgs()) {
          sql_sig.add(toSqlTypeName(arg_type).getFamily());
        }
        return sql_sig;
      }

      @Override
      public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
        final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
        return typeFactory.createSqlType(ret);
      }

      private static SqlTypeName toSqlTypeName(final ExtensionFunction.ExtArgumentType type) {
        switch (type) {
          case Int16:
            return SqlTypeName.SMALLINT;
          case Int32:
            return SqlTypeName.INTEGER;
          case Int64:
            return SqlTypeName.BIGINT;
          case Float:
            return SqlTypeName.FLOAT;
          case Double:
            return SqlTypeName.DOUBLE;
        }
        assert false;
        return null;
      }

      private final SqlTypeName ret;
    }
}

// End MapDSqlOperatorTable.java

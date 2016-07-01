/*
 * Clever MapD license
 */
package com.mapd.calcite.parser;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.prepare.Prepare;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeComparability;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rel.type.RelDataTypeFieldImpl;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.ObjectSqlType;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.validate.SqlMoniker;
import org.apache.calcite.sql.validate.SqlMonikerImpl;
import org.apache.calcite.sql.validate.SqlMonikerType;
import org.apache.calcite.sql.validate.SqlValidatorUtil;
import org.apache.calcite.util.Util;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.mapd.metadata.MetaConnect;
import com.mapd.thrift.server.TColumnType;
import com.mapd.thrift.server.TDatumType;
import com.mapd.thrift.server.TTypeInfo;
import java.util.ArrayList;

import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * MapD Catalog reader Includes default SALES schema for testing purposes
 */
public class MapDCatalogReader implements Prepare.CatalogReader {
  //~ Static fields/initializers ---------------------------------------------

  final static Logger MAPDLOGGER = LoggerFactory.getLogger(MapDCatalogReader.class);

  protected static final String DEFAULT_CATALOG = "CATALOG";
  protected String CURRENT_DEFAULT_SCHEMA = "mapd";

  private static volatile Map<List<String>, MapDTable> MAPD_TABLES = Maps.newConcurrentMap();
  private static volatile Map<String, MapDDatabase> MAPD_DATABASE = Maps.newConcurrentMap();

  //~ Instance fields --------------------------------------------------------
  protected final RelDataTypeFactory typeFactory;
  private final boolean elideRecord = true;
  private RelDataType addressType;
  private boolean caseSensitive = false;
  private final EnumMap<TDatumType, ArrayList<ArrayList<RelDataType>>> mapDTypes;
  private MapDUser currentMapDUser;
  private final String dataDir;

  //~ Constructors -----------------------------------------------------------
  /**
   * Creates a MapDCatalogReader.
   *
   * <p>
   * Caller must then call {@link #init} to populate with data.</p>
   *
   * @param typeFactory Type factory
   * @param dataDir directory containing the mapd data
   *
   */
  public MapDCatalogReader(RelDataTypeFactory typeFactory, String dataDir) {
    this.typeFactory = typeFactory;
    this.dataDir = dataDir;

    // add all the MapD datatype into this structure
    // it is indexed with the TDatumType,  isArray , isNullable
    mapDTypes = new EnumMap<TDatumType, ArrayList<ArrayList<RelDataType>>>(TDatumType.class);

    for (TDatumType dType : TDatumType.values()) {
      RelDataType cType = getRelDataType(dType);
      ArrayList<ArrayList<RelDataType>> nullList = new ArrayList<ArrayList<RelDataType>>(2);
      for (int nullable = 0; nullable < 2; nullable++) {
        ArrayList<RelDataType> arrayList = new ArrayList<RelDataType>(2);
        if (nullable == 0) {
          arrayList.add(0, cType);                                              // regular type
          arrayList.add(1, typeFactory.createArrayType(cType, -1));             // Array type
        } else {
          arrayList.add(0, typeFactory.createTypeWithNullability(cType, true)); // regular type nullable
          arrayList.add(1, typeFactory.createArrayType(arrayList.get(0), -1));  // Array type nullable
        }
        nullList.add(nullable, arrayList);
      }
      mapDTypes.put(dType, nullList);
    }

    addDefaultTestSchemas();
  }

  private MapDTable getTableData(String tableName) {

    MetaConnect metaConnect = new MetaConnect(dataDir, currentMapDUser.getDB());
    metaConnect.connectToDBCatalog();
    // Now get tables column details
    Map<String, TColumnType> tableDescriptor = metaConnect.getTableDescriptor(tableName);

    // get database
    MapDDatabase db = MAPD_DATABASE.get(currentMapDUser.getDB());
    // if schema doesn't exist create it and store it
    // note we are in sync block here as all table create is managed in sync
    if (db == null) {
      db = new MapDDatabase(currentMapDUser.getDB());
      registerSchema(db);
    }

    MAPDLOGGER.debug("Database is " + currentMapDUser.getDB());

    MAPDLOGGER.debug("\t table  is " + tableName);
    MapDTable mtable = MapDTable.create(this, db, tableName, false);

    // if we have a table descriptor from mapd server
    if (tableDescriptor != null) {
      for (Map.Entry<String, TColumnType> entry : tableDescriptor.entrySet()) {
        TColumnType value = entry.getValue();
        MAPDLOGGER.debug("'" + entry.getKey() + "'"
                + " \t" + value.getCol_type().getEncoding()
                + " \t" + value.getCol_type().getFieldValue(TTypeInfo._Fields.TYPE)
                + " \t" + value.getCol_type().nullable
                + " \t" + value.getCol_type().is_array
        );

        mtable.addColumn(entry.getKey(), mapDTypes.get(value.getCol_type().type)
                .get(value.getCol_type().nullable ? 1 : 0)
                .get(value.getCol_type().is_array ? 1 : 0));

      }
      registerTable(mtable);
    } else {
      // no table in MapD server schema
      mtable = null;
    }
    return mtable;
  }

  /**
   * This schema adds for testing purposes the scott/tiger default schema
   */
  private MapDCatalogReader addDefaultTestSchemas() {

    final RelDataType intType
            = typeFactory.createSqlType(SqlTypeName.INTEGER);
    final RelDataType intTypeNull
            = typeFactory.createTypeWithNullability(intType, true);
    final RelDataType varchar10Type
            = typeFactory.createSqlType(SqlTypeName.VARCHAR, 10);
    final RelDataType varchar20Type
            = typeFactory.createSqlType(SqlTypeName.VARCHAR, 20);
    final RelDataType timestampType
            = typeFactory.createSqlType(SqlTypeName.TIMESTAMP);
    final RelDataType stringArrayType
            = typeFactory.createArrayType(varchar10Type, -1);
    final RelDataType booleanType
            = typeFactory.createSqlType(SqlTypeName.BOOLEAN);
    final RelDataType rectilinearCoordType
            = typeFactory.builder().add("X", intType).add("Y", intType).build();

    // TODO jvs 12-Feb-2005: register this canonical instance with type
    // factory
    addressType
            = new ObjectSqlType(
                    SqlTypeName.STRUCTURED,
                    new SqlIdentifier("ADDRESS", SqlParserPos.ZERO),
                    false,
                    Arrays.asList(
                            new RelDataTypeFieldImpl("STREET", 0, varchar20Type),
                            new RelDataTypeFieldImpl("CITY", 1, varchar20Type),
                            new RelDataTypeFieldImpl("ZIP", 1, intType),
                            new RelDataTypeFieldImpl("STATE", 1, varchar20Type)),
                    RelDataTypeComparability.NONE);

    // Register "SALES" schema.
    MapDDatabase salesSchema = new MapDDatabase("SALES");
    registerSchema(salesSchema);

    // Register "EMP" table.
    final MapDTable empTable
            = MapDTable.create(this, salesSchema, "EMP", false);
    empTable.addColumn("EMPNO", intType);
    empTable.addColumn("ENAME", varchar20Type);
    empTable.addColumn("JOB", varchar10Type);
    empTable.addColumn("MGR", intTypeNull);
    empTable.addColumn("HIREDATE", timestampType);
    empTable.addColumn("SAL", intType);
    empTable.addColumn("COMM", intType);
    empTable.addColumn("DEPTNO", intType);
    empTable.addColumn("SLACKER", booleanType);
    empTable.addColumn("SLACKARR1", stringArrayType);
    empTable.addColumn("SLACKARR2", stringArrayType);
    registerTable(empTable);

    // Register "DEPT" table.
    MapDTable deptTable = MapDTable.create(this, salesSchema, "DEPT", false);
    deptTable.addColumn("DEPTNO", intType);
    deptTable.addColumn("NAME", varchar10Type);
    registerTable(deptTable);

    // Register "BONUS" table.
    MapDTable bonusTable = MapDTable.create(this, salesSchema, "BONUS", false);
    bonusTable.addColumn("ENAME", varchar20Type);
    bonusTable.addColumn("JOB", varchar10Type);
    bonusTable.addColumn("SAL", intType);
    bonusTable.addColumn("COMM", intType);
    registerTable(bonusTable);

    // Register "SALGRADE" table.
    MapDTable salgradeTable = MapDTable.create(this, salesSchema, "SALGRADE",
            false);
    salgradeTable.addColumn("GRADE", intType);
    salgradeTable.addColumn("LOSAL", intType);
    salgradeTable.addColumn("HISAL", intType);
    registerTable(salgradeTable);

    // Register "EMP_ADDRESS" table
    MapDTable contactAddressTable
            = MapDTable.create(this, salesSchema, "EMP_ADDRESS", false);
    contactAddressTable.addColumn("EMPNO", intType);
    contactAddressTable.addColumn("HOME_ADDRESS", addressType);
    contactAddressTable.addColumn("MAILING_ADDRESS", addressType);
    registerTable(contactAddressTable);

    // Register "CUSTOMER" schema.
    MapDDatabase customerSchema = new MapDDatabase("CUSTOMER");
    registerSchema(customerSchema);

    // Register "CONTACT" table.
    MapDTable contactTable = MapDTable.create(this, customerSchema, "CONTACT",
            false);
    contactTable.addColumn("CONTACTNO", intType);
    contactTable.addColumn("FNAME", varchar10Type);
    contactTable.addColumn("LNAME", varchar10Type);
    contactTable.addColumn("EMAIL", varchar20Type);
    contactTable.addColumn("COORD", rectilinearCoordType);
    registerTable(contactTable);

    // Register "ACCOUNT" table.
    MapDTable accountTable = MapDTable.create(this, customerSchema, "ACCOUNT",
            false);
    accountTable.addColumn("ACCTNO", intType);
    accountTable.addColumn("TYPE", varchar20Type);
    accountTable.addColumn("BALANCE", intType);
    registerTable(accountTable);

    // Register "ORDERS" stream.
    MapDTable ordersStream = MapDTable.create(this, salesSchema, "ORDERS",
            true);
    ordersStream.addColumn("ROWTIME", timestampType);
    ordersStream.addMonotonic("ROWTIME");
    ordersStream.addColumn("PRODUCTID", intType);
    ordersStream.addColumn("ORDERID", intType);
    registerTable(ordersStream);

    // Register "SHIPMENTS" stream.
    MapDTable shipmentsStream = MapDTable.create(this, salesSchema, "SHIPMENTS",
            true);
    shipmentsStream.addColumn("ROWTIME", timestampType);
    shipmentsStream.addMonotonic("ROWTIME");
    shipmentsStream.addColumn("ORDERID", intType);
    registerTable(shipmentsStream);

    return this;
  }

  //~ Methods ----------------------------------------------------------------
  /**
   *
   * @param opName
   * @param category
   * @param syntax
   * @param operatorList
   */
  @Override
  public void lookupOperatorOverloads(SqlIdentifier opName,
          SqlFunctionCategory category, SqlSyntax syntax,
          List<SqlOperator> operatorList) {
  }

  /**
   *
   * @return
   */
  @Override
  public List<SqlOperator> getOperatorList() {
    return ImmutableList.of();
  }

  /**
   *
   * @param schemaPath
   * @return
   */
  @Override
  public Prepare.CatalogReader withSchemaPath(List<String> schemaPath) {
    return this;
  }

  /**
   *
   * @param names
   * @return
   */
  @Override
  public Prepare.PreparingTable getTableForMember(List<String> names) {
    return getTable(names);
  }

  /**
   *
   * @return
   */
  @Override
  public RelDataTypeFactory getTypeFactory() {
    return typeFactory;
  }

  /**
   *
   * @param planner
   */
  @Override
  public void registerRules(RelOptPlanner planner) {
  }

  protected void registerTable(MapDTable table) {
    table.onRegister(typeFactory);
    List<String> names = table.getQualifiedName();
    MAPD_TABLES.put(
            ImmutableList.of(names.get(0).toUpperCase(), names.get(1).toUpperCase(), names.get(2).toUpperCase()),
            table);
  }

  protected void registerSchema(MapDDatabase schema) {
    MAPD_DATABASE.put(schema.getSchemaName(), schema);
  }

  /**
   *
   * @param names
   * @return
   */
  @Override
  public Prepare.PreparingTable getTable(final List<String> names) {
    switch (names.size()) {
      case 1:
        return getMapDTable(
                ImmutableList.of(DEFAULT_CATALOG, this.currentMapDUser.getDB().toUpperCase(), names.get(0).toUpperCase()));
      case 2:
        return getMapDTable(
                ImmutableList.of(DEFAULT_CATALOG, names.get(0).toUpperCase(), names.get(1).toUpperCase()));
      case 3:
        return getMapDTable(
                ImmutableList.of(names.get(0).toUpperCase(), names.get(1).toUpperCase(), names.get(2).toUpperCase()));
      default:
        return null;
    }
  }

  private MapDTable getMapDTable(List<String> names) {
    // get the mapd table if we have it in map
    // if not see if it exists and add it to list and then return it
    MapDTable returnTable = MAPD_TABLES.get(names);

    // in case a table doesn't exist in map check it has not been added
    // so check the mapd server for the new table
    if (returnTable == null) {
      synchronized (this) {
        returnTable = MAPD_TABLES.get(names);
        if (returnTable == null) {
          returnTable = getTableData(names.get(2));
        }
      }
    }
    return returnTable;
  }

  /**
   *
   * @param typeName
   * @return
   */
  @Override
  public RelDataType getNamedType(SqlIdentifier typeName) {
    if (typeName.equalsDeep(
            addressType.getSqlIdentifier(),
            false)) {
      return addressType;
    } else {
      return null;
    }
  }

  /**
   *
   * @param names
   * @return
   */
  @Override
  public List<SqlMoniker> getAllSchemaObjectNames(List<String> names) {
    List<SqlMoniker> result;
    switch (names.size()) {
      case 0:
        // looking for catalog and schema names
        return ImmutableList.<SqlMoniker>builder()
                .add(new SqlMonikerImpl(DEFAULT_CATALOG, SqlMonikerType.CATALOG))
                .addAll(getAllSchemaObjectNames(ImmutableList.of(DEFAULT_CATALOG)))
                .build();
      case 1:
        // looking for schema names
        result = Lists.newArrayList();
        for (MapDDatabase schema : MAPD_DATABASE.values()) {
          final String catalogName = names.get(0);
          if (schema.getCatalogName().equals(catalogName)) {
            final ImmutableList<String> names1
                    = ImmutableList.of(catalogName, schema.getSchemaName());
            result.add(new SqlMonikerImpl(names1, SqlMonikerType.SCHEMA));
          }
        }
        return result;
      case 2:
        // looking for table names in the given schema
        MapDDatabase schema = MAPD_DATABASE.get(names.get(1));
        if (schema == null) {
          return Collections.emptyList();
        }
        result = Lists.newArrayList();
        for (String tableName : schema.getTableNames()) {
          result.add(
                  new SqlMonikerImpl(
                          ImmutableList.of(schema.getCatalogName(), schema.getSchemaName(),
                                  tableName),
                          SqlMonikerType.TABLE));
        }
        return result;
      default:
        return Collections.emptyList();
    }
  }

  @Override
  public List<String> getSchemaName() {
    return ImmutableList.of(DEFAULT_CATALOG, CURRENT_DEFAULT_SCHEMA);
  }

  @Override
  public RelDataTypeField field(RelDataType rowType, String alias) {
    return SqlValidatorUtil.lookupField(caseSensitive, elideRecord, rowType,
            alias);
  }

  @Override
  public int fieldOrdinal(RelDataType rowType, String alias) {
    final RelDataTypeField field = field(rowType, alias);
    return field != null ? field.getIndex() : -1;
  }

  @Override
  public boolean matches(String string, String name) {
    MAPDLOGGER.debug("matches  " + string + " vs " + name);
    return Util.matches(caseSensitive, string, name);
  }

  @Override
  public int match(List<String> strings, String name) {
    MAPDLOGGER.debug("matches  " + strings + " vs " + name);
    return Util.findMatch(strings, name, caseSensitive);
  }

  @Override
  public RelDataType createTypeFromProjection(final RelDataType type,
          final List<String> columnNameList) {
    return SqlValidatorUtil.createTypeFromProjection(type, columnNameList,
            typeFactory, caseSensitive, elideRecord);
  }

  public void setCurrentMapDUser(MapDUser mapDUser) {
    currentMapDUser = mapDUser;
  }

  // Convert our TDataumn type in to a base calcite SqlType
  // todo confirm whether it is ok to ignore thinsg like lengths
  // since we do not use them on the validator side of the calcite 'fence'
  private RelDataType getRelDataType(TDatumType dType) {

    switch (dType) {
      case SMALLINT:
        return typeFactory.createSqlType(SqlTypeName.SMALLINT);
      case INT:
        return typeFactory.createSqlType(SqlTypeName.INTEGER);
      case BIGINT:
        return typeFactory.createSqlType(SqlTypeName.BIGINT);
      case FLOAT:
        return typeFactory.createSqlType(SqlTypeName.FLOAT);
      case DECIMAL:
        return typeFactory.createSqlType(SqlTypeName.DECIMAL);
      case DOUBLE:
        return typeFactory.createSqlType(SqlTypeName.DOUBLE);
      case STR:
        return typeFactory.createSqlType(SqlTypeName.VARCHAR, 50);
      case TIME:
        return typeFactory.createSqlType(SqlTypeName.TIME);
      case TIMESTAMP:
        return typeFactory.createSqlType(SqlTypeName.TIMESTAMP);
      case DATE:
        return typeFactory.createSqlType(SqlTypeName.DATE);
      case BOOL:
        return typeFactory.createSqlType(SqlTypeName.BOOLEAN);
      default:
        throw new AssertionError(dType.name());
    }
  }
}
// End MapDCatalogReader.java

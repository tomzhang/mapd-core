enum TDatumType {
  INT,
  REAL,
  STR,
  TIME,
  TIMESTAMP,
  DATE,
  BOOL
}

enum TExecuteMode {
  HYBRID,
  GPU,
  CPU
}

union TDatum {
  1: i64 int_val,
  2: double real_val,
  3: string str_val
}

struct TColumnValue {
  1: TDatum datum,
  2: bool is_null
}

struct TTypeInfo {
  1: TDatumType type,
  2: bool nullable
}

struct TColumnType {
  1: string col_name,
  2: TTypeInfo col_type
}

struct TRow {
  1: list<TColumnValue> cols
}

struct TRowSet {
  1: TRowDescriptor row_desc
  2: list<TRow> rows
}

typedef list<TColumnType> TRowDescriptor
typedef map<string, TColumnType> TTableDescriptor
typedef i32 TSessionId
typedef byte TLoadId

struct TQueryResult {
  1: TRowSet row_set
  2: i64 execution_time_ms
}

struct TDBInfo {
  1: string db_name
  2: string db_owner
}

exception TMapDException {
  1: string error_msg
}

exception ThriftException {
  1: string error_msg
}

service MapD {
  TSessionId connect(1: string user, 2: string passwd, 3: string dbname) throws (1: TMapDException e 2: ThriftException te)
  void disconnect(1: TSessionId session) throws (1: TMapDException e 2: ThriftException te)
  TQueryResult sql_execute(1: TSessionId session, 2: string query) throws (1: TMapDException e 2: ThriftException te)
  TTableDescriptor get_table_descriptor(1: TSessionId session, 2: string table_name) throws (1: TMapDException e 2: ThriftException te)
  list<string> get_tables(1: TSessionId session) throws (1: TMapDException e 2: ThriftException te)
  list<string> get_users() throws (1: ThriftException te)
  list<TDBInfo> get_databases() throws (1: ThriftException te)
  void set_execution_mode(1: TExecuteMode mode) throws (1: TMapDException e 2: ThriftException te)
  string get_version() throws (1: ThriftException te)
  TLoadId start_load(1: TSessionId session, 2: string table_name) throws (1: TMapDException e 2: ThriftException te)
  void load_table(1: TSessionId session, 2: TLoadId load, 3: TRowSet rows) throws (1: TMapDException e 2: ThriftException te)
  void end_load(1: TSessionId session, 2: TLoadId load) throws (1: TMapDException e 2: ThriftException te)
}

service RemoteStringDictionary {
  void create(1: i32 dict_id, 2: i32 db_id)
  i32 get(1: string str, 2: i32 dict_id)
  string get_string(1: i32 string_id, 2: i32 dict_id)
  i64 storage_entry_count(1: i32 dict_id)
}

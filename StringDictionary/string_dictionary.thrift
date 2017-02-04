service RemoteStringDictionary {
  void create(1: i32 dict_id, 2: i32 db_id)
  i32 get(1: string str, 2: i32 dict_id)
  string get_string(1: i32 string_id, 2: i32 dict_id)
  i64 storage_entry_count(1: i32 dict_id)
  list<string> get_like(1: string pattern, 2: bool icase, 3: bool is_simple, 4: string escape, 5: i64 generation, 6: i32 dict_id)
  list<string> get_regexp_like(1: string pattern, 2: string escape, 3: i64 generation, 4: i32 dict_id)
}

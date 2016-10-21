#include "QueryRunner.h"

#include "../Parser/parser.h"
#include "../SqliteConnector/SqliteConnector.h"
#include "../Import/Importer.h"

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace std;

namespace {

std::unique_ptr<Catalog_Namespace::SessionInfo> g_session;
bool g_hoist_literals{true};

ResultRows run_multiple_agg(const string& query_str, const ExecutorDeviceType device_type) {
  return run_multiple_agg(query_str, g_session, device_type, g_hoist_literals);
}

TargetValue run_simple_agg(const string& query_str, const ExecutorDeviceType device_type) {
  auto rows = run_multiple_agg(query_str, device_type);
  auto crt_row = rows.getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size());
  return crt_row[0];
}

template <class T>
T v(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  CHECK(scalar_r);
  auto p = boost::get<T>(scalar_r);
  CHECK(p);
  return *p;
}

void run_ddl_statement(const string& create_table_stmt) {
  SQLParser parser;
  list<std::unique_ptr<Parser::Stmt>> parse_trees;
  string last_parsed;
  CHECK_EQ(parser.parse(create_table_stmt, parse_trees, last_parsed), 0);
  CHECK_EQ(parse_trees.size(), size_t(1));
  auto stmt = parse_trees.front().get();
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
  CHECK(ddl);
  if (ddl != nullptr)
    ddl->execute(*g_session);
}

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !g_session->get_catalog().get_dataMgr().gpusPresent();
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

bool approx_eq(const double v, const double target, const double eps = 0.01) {
  const auto v_u64 = *reinterpret_cast<const uint64_t*>(&v);
  const auto target_u64 = *reinterpret_cast<const uint64_t*>(&target);
  return v_u64 == target_u64 || (target - eps < v && v < target + eps);
}

class SQLiteComparator {
 public:
  SQLiteComparator() : connector_("main", "") {}

  void query(const std::string& query_string) { connector_.query(query_string); }

  void compare(const std::string& query_string, const ExecutorDeviceType device_type) {
    compare_impl(query_string, device_type, false);
  }

  // added to deal with time shift for now testing
  void compare_timstamp_approx(const std::string& query_string, const ExecutorDeviceType device_type) {
    compare_impl(query_string, device_type, true);
  }

  void compare_impl(const std::string& query_string, const ExecutorDeviceType device_type, bool timestamp_approx) {
    connector_.query(query_string);
    const auto mapd_results = run_multiple_agg(query_string, device_type);
    ASSERT_EQ(connector_.getNumRows(), mapd_results.rowCount());
    const int num_rows{static_cast<int>(connector_.getNumRows())};
    if (mapd_results.definitelyHasNoRows()) {
      ASSERT_EQ(0, num_rows);
      return;
    }
    if (!num_rows) {
      return;
    }
    CHECK_EQ(connector_.getNumCols(), mapd_results.colCount());
    const int num_cols{static_cast<int>(connector_.getNumCols())};
    for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
      const auto crt_row = mapd_results.getNextRow(true, true);
      CHECK(!crt_row.empty());
      CHECK_EQ(static_cast<size_t>(num_cols), crt_row.size());
      for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
        const auto ref_col_type = connector_.columnTypes[col_idx];
        const auto mapd_variant = crt_row[col_idx];
        const auto scalar_mapd_variant = boost::get<ScalarTargetValue>(&mapd_variant);
        CHECK(scalar_mapd_variant);
        const auto mapd_ti = mapd_results.getColType(col_idx);
        const auto mapd_type = mapd_ti.get_type();
        checkTypeConsistency(ref_col_type, mapd_ti);
        const bool ref_is_null = connector_.isNull(row_idx, col_idx);
        switch (mapd_type) {
          case kSMALLINT:
          case kINT:
          case kBIGINT: {
            const auto mapd_as_int_p = boost::get<int64_t>(scalar_mapd_variant);
            ASSERT_NE(nullptr, mapd_as_int_p);
            const auto mapd_val = *mapd_as_int_p;
            if (ref_is_null) {
              ASSERT_EQ(inline_int_null_val(mapd_ti), mapd_val);
            } else {
              const auto ref_val = connector_.getData<int64_t>(row_idx, col_idx);
              ASSERT_EQ(ref_val, mapd_val);
            }
            break;
          }
          case kTEXT:
          case kVARCHAR: {
            const auto mapd_as_str_p = boost::get<NullableString>(scalar_mapd_variant);
            ASSERT_NE(nullptr, mapd_as_str_p);
            const auto mapd_str_notnull = boost::get<std::string>(mapd_as_str_p);
            if (ref_is_null) {
              CHECK(!mapd_str_notnull);
            } else {
              CHECK(mapd_str_notnull);
              const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
              const auto mapd_val = *mapd_str_notnull;
              ASSERT_EQ(ref_val, mapd_val);
            }
            break;
          }
          case kNUMERIC:
          case kDECIMAL:
          case kDOUBLE: {
            const auto mapd_as_double_p = boost::get<double>(scalar_mapd_variant);
            ASSERT_NE(nullptr, mapd_as_double_p);
            const auto mapd_val = *mapd_as_double_p;
            if (ref_is_null) {
              ASSERT_EQ(inline_fp_null_val(SQLTypeInfo(kDOUBLE, false)), mapd_val);
            } else {
              const auto ref_val = connector_.getData<double>(row_idx, col_idx);
              ASSERT_TRUE(approx_eq(ref_val, mapd_val));
            }
            break;
          }
          case kFLOAT: {
            const auto mapd_as_float_p = boost::get<float>(scalar_mapd_variant);
            ASSERT_NE(nullptr, mapd_as_float_p);
            const auto mapd_val = *mapd_as_float_p;
            if (ref_is_null) {
              if (inline_fp_null_val(SQLTypeInfo(kFLOAT, false)) != mapd_val) {
                CHECK(false);
              }
            } else {
              const auto ref_val = connector_.getData<float>(row_idx, col_idx);
              if (!approx_eq(ref_val, mapd_val)) {
                CHECK(false);
              }
            }
            break;
          }
          case kTIMESTAMP:
          case kDATE: {
            const auto mapd_as_int_p = boost::get<int64_t>(scalar_mapd_variant);
            CHECK(mapd_as_int_p);
            const auto mapd_val = *mapd_as_int_p;
            if (ref_is_null) {
              CHECK_EQ(inline_int_null_val(mapd_ti), mapd_val);
            } else {
              struct tm tm_struct {
                0
              };
              const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
              const auto end_str =
                  strptime(ref_val.c_str(), mapd_type == kTIMESTAMP ? "%Y-%m-%d %H:%M:%S" : "%Y-%m-%d", &tm_struct);
              if (end_str != nullptr) {
                ASSERT_EQ(0, *end_str);
                ASSERT_EQ(ref_val.size(), static_cast<size_t>(end_str - ref_val.c_str()));
              }
              if (timestamp_approx) {
                // approximate result give 10 second lee way
                ASSERT_NEAR(*mapd_as_int_p, timegm(&tm_struct), 10);
              } else {
                ASSERT_EQ(*mapd_as_int_p, timegm(&tm_struct));
              }
            }
            break;
          }
          case kBOOLEAN: {
            const auto mapd_as_int_p = boost::get<int64_t>(scalar_mapd_variant);
            CHECK(mapd_as_int_p);
            const auto mapd_val = *mapd_as_int_p;
            if (ref_is_null) {
              CHECK_EQ(inline_int_null_val(mapd_ti), mapd_val);
            } else {
              const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
              if (ref_val == "t") {
                ASSERT_EQ(1, *mapd_as_int_p);
              } else {
                CHECK_EQ("f", ref_val);
                ASSERT_EQ(0, *mapd_as_int_p);
              }
            }
            break;
          }
          case kTIME: {
            const auto mapd_as_int_p = boost::get<int64_t>(scalar_mapd_variant);
            CHECK(mapd_as_int_p);
            const auto mapd_val = *mapd_as_int_p;
            if (ref_is_null) {
              CHECK_EQ(inline_int_null_val(mapd_ti), mapd_val);
            } else {
              const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
              std::vector<std::string> time_tokens;
              boost::split(time_tokens, ref_val, boost::is_any_of(":"));
              ASSERT_EQ(size_t(3), time_tokens.size());
              ASSERT_EQ(boost::lexical_cast<int64_t>(time_tokens[0]) * 3600 +
                            boost::lexical_cast<int64_t>(time_tokens[1]) * 60 +
                            boost::lexical_cast<int64_t>(time_tokens[2]),
                        *mapd_as_int_p);
            }
            break;
          }
          default:
            CHECK(false);
        }
      }
    }
  }

 private:
  static void checkTypeConsistency(const int ref_col_type, const SQLTypeInfo& mapd_ti) {
    if (ref_col_type == SQLITE_NULL) {
      // TODO(alex): re-enable the check that mapd_ti is nullable,
      //             got invalidated because of outer joins
      return;
    }
    if (mapd_ti.is_integer()) {
      CHECK_EQ(SQLITE_INTEGER, ref_col_type);
    } else if (mapd_ti.is_fp() || mapd_ti.is_decimal()) {
      CHECK_EQ(SQLITE_FLOAT, ref_col_type);
    } else {
      CHECK_EQ(SQLITE_TEXT, ref_col_type);
    }
  }

  SqliteConnector connector_;
};

const ssize_t g_num_rows{10};
SQLiteComparator g_sqlite_comparator;

void c(const std::string& query_string, const ExecutorDeviceType device_type) {
  g_sqlite_comparator.compare(query_string, device_type);
}

/* timestamp approximate checking for NOW() */
void cta(const std::string& query_string, const ExecutorDeviceType device_type) {
  g_sqlite_comparator.compare_timstamp_approx(query_string, device_type);
}
}  // namespace

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

TEST(Select, FilterAndSimpleAggregation) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test;", dt);
    c("SELECT COUNT(f) FROM test;", dt);
    c("SELECT MIN(x) FROM test;", dt);
    c("SELECT MAX(x) FROM test;", dt);
    c("SELECT MIN(z) FROM test;", dt);
    c("SELECT MAX(z) FROM test;", dt);
    c("SELECT MIN(t) FROM test;", dt);
    c("SELECT MAX(t) FROM test;", dt);
    c("SELECT SUM(x + y) FROM test;", dt);
    c("SELECT SUM(x + y + z) FROM test;", dt);
    c("SELECT SUM(x + y + z + t) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8;", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102;", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 OR (z > 100 AND z < 103);", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > 1000 AND t < 1002;", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 OR (z > 100 AND z < 103);", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 OR (z > 100 AND z < 102) OR (t > 1000 AND t < 1003);", dt);
    c("SELECT COUNT(*) FROM test WHERE x <> 7;", dt);
    c("SELECT COUNT(*) FROM test WHERE z <> 102;", dt);
    c("SELECT COUNT(*) FROM test WHERE t <> 1002;", dt);
    c("SELECT COUNT(*) FROM test WHERE x + y = 49;", dt);
    c("SELECT COUNT(*) FROM test WHERE x + y + z = 150;", dt);
    c("SELECT COUNT(*) FROM test WHERE x + y + z + t = 1151;", dt);
    c("SELECT SUM(x + y) FROM test WHERE x + y = 49;", dt);
    c("SELECT SUM(x + y + z) FROM test WHERE x + y = 49;", dt);
    c("SELECT SUM(x + y + z + t) FROM test WHERE x + y = 49;", dt);
    c("SELECT COUNT(*) FROM test WHERE x - y = -35;", dt);
    c("SELECT COUNT(*) FROM test WHERE x - y + z = 66;", dt);
    c("SELECT COUNT(*) FROM test WHERE x - y + z + t = 1067;", dt);
    c("SELECT COUNT(*) FROM test WHERE y - x = 35;", dt);
    c("SELECT SUM(2 * x) FROM test WHERE x = 7;", dt);
    c("SELECT SUM(2 * x + z) FROM test WHERE x = 7;", dt);
    c("SELECT SUM(x + y) FROM test WHERE x - y = -35;", dt);
    c("SELECT SUM(x + y) FROM test WHERE y - x = 35;", dt);
    c("SELECT SUM(x + y - z) FROM test WHERE y - x = 35;", dt);
    c("SELECT SUM(x * y + 15) FROM test WHERE x + y + 1 = 50;", dt);
    c("SELECT SUM(x * y + 15) FROM test WHERE x + y + z + 1 = 151;", dt);
    c("SELECT SUM(x * y + 15) FROM test WHERE x + y + z + t + 1 = 1152;", dt);
    c("SELECT MIN(x * y + 15) FROM test WHERE x + y + 1 = 50;", dt);
    c("SELECT MIN(x * y + 15) FROM test WHERE x + y + z + 1 = 151;", dt);
    c("SELECT MIN(x * y + 15) FROM test WHERE x + y + z + t + 1 = 1152;", dt);
    c("SELECT MAX(x * y + 15) FROM test WHERE x + y + 1 = 50;", dt);
    c("SELECT MAX(x * y + 15) FROM test WHERE x + y + z + 1 = 151;", dt);
    c("SELECT MAX(x * y + 15) FROM test WHERE x + y + z + t + 1 = 1152;", dt);
    c("SELECT MIN(x) FROM test WHERE x = 7;", dt);
    c("SELECT MIN(z) FROM test WHERE z = 101;", dt);
    c("SELECT MIN(t) FROM test WHERE t = 1001;", dt);
    c("SELECT AVG(x + y) FROM test;", dt);
    c("SELECT AVG(x + y + z) FROM test;", dt);
    c("SELECT AVG(x + y + z + t) FROM test;", dt);
    c("SELECT AVG(y) FROM test WHERE x > 6 AND x < 8;", dt);
    c("SELECT AVG(y) FROM test WHERE z > 100 AND z < 102;", dt);
    c("SELECT AVG(y) FROM test WHERE t > 1000 AND t < 1002;", dt);
    c("SELECT MIN(dd) FROM test;", dt);
    c("SELECT MAX(dd) FROM test;", dt);
    c("SELECT SUM(dd) FROM test;", dt);
    c("SELECT AVG(dd) FROM test;", dt);
    c("SELECT AVG(dd) FROM test WHERE x > 6 AND x < 8;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 100;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 200;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 300;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 111.0;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 111.1;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 222.2;", dt);
    c("SELECT MAX(x + dd) FROM test;", dt);
    c("SELECT MAX(x + 2 * dd), MIN(x + 2 * dd) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > CAST(111.0 AS decimal(10, 2));", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > CAST(222.0 AS decimal(10, 2));", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > CAST(333.0 AS decimal(10, 2));", dt);
    c("SELECT MIN(dd * dd) FROM test;", dt);
    c("SELECT MAX(dd * dd) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE u IS NOT NULL;", dt);
    c("SELECT AVG(u * f) FROM test;", dt);
    c("SELECT AVG(u * d) FROM test;", dt);
    c("SELECT SUM(-y) FROM test;", dt);
    c("SELECT SUM(-z) FROM test;", dt);
    c("SELECT SUM(-t) FROM test;", dt);
    c("SELECT SUM(-dd) FROM test;", dt);
    c("SELECT SUM(-f) FROM test;", dt);
    c("SELECT SUM(-d) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE ofq >= 0 OR ofq IS NULL;", dt);
    c("SELECT COUNT(*) AS val FROM test WHERE (test.dd = 0.5 OR test.dd = 3);", dt);
#ifdef HAVE_CALCITE
    c("SELECT COUNT(*) FROM test WHERE d = 2.2", dt);
    c("SELECT COUNT(*) FROM test WHERE fx + 1 IS NULL;", dt);
    c("SELECT COUNT(ss) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE null IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE null IS NOT NULL;", dt);
#endif  // HAVE_CALCITE
    ASSERT_EQ(15, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE MOD(x, 7) = 0;", dt)));
    ASSERT_EQ(0, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE MOD(x, 7) = 7;", dt)));
    ASSERT_EQ(5, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE MOD(x, 7) <> 0;", dt)));
    ASSERT_EQ(20, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE MOD(x, 7) <> 7;", dt)));
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x) FROM test WHERE x <> 7 AND x <> 8;", dt)),
              numeric_limits<int64_t>::max());
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x) FROM test WHERE z <> 101 AND z <> 102;", dt)),
              numeric_limits<int64_t>::max());
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x) FROM test WHERE t <> 1001 AND t <> 1002;", dt)),
              numeric_limits<int64_t>::max());
  }
}

TEST(Select, LimitAndOffset) {
  CHECK(g_num_rows >= 4);
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      const auto rows = run_multiple_agg("SELECT * FROM test LIMIT 5;", dt);
      ASSERT_EQ(size_t(5), rows.rowCount());
    }
    {
      const auto rows = run_multiple_agg("SELECT * FROM test LIMIT 5 OFFSET 3;", dt);
      ASSERT_EQ(size_t(5), rows.rowCount());
    }
    {
      const auto rows = run_multiple_agg("SELECT * FROM test WHERE x <> 8 LIMIT 3 OFFSET 1;", dt);
      ASSERT_EQ(size_t(3), rows.rowCount());
    }
  }
}

TEST(Select, FloatAndDoubleTests) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT MIN(f) FROM test;", dt);
    c("SELECT MAX(f) FROM test;", dt);
    c("SELECT AVG(f) FROM test;", dt);
    c("SELECT MIN(d) FROM test;", dt);
    c("SELECT MAX(d) FROM test;", dt);
    c("SELECT AVG(d) FROM test;", dt);
    c("SELECT SUM(f) FROM test;", dt);
    c("SELECT SUM(d) FROM test;", dt);
    c("SELECT SUM(f + d) FROM test;", dt);
    c("SELECT AVG(x * f) FROM test;", dt);
    c("SELECT AVG(z - 200) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE f > 1.0 AND f < 1.2;", dt);
    c("SELECT COUNT(*) FROM test WHERE f > 1.101 AND f < 1.299;", dt);
    c("SELECT COUNT(*) FROM test WHERE f > 1.201 AND f < 1.4;", dt);
    c("SELECT COUNT(*) FROM test WHERE f > 1.0 AND f < 1.2 AND d > 2.0 AND d < 2.4;", dt);
    c("SELECT COUNT(*) FROM test WHERE f > 1.0 AND f < 1.2 OR (d > 2.0 AND d < 3.0);", dt);
    c("SELECT SUM(x + y) FROM test WHERE f > 1.0 AND f < 1.2;", dt);
    c("SELECT SUM(x + y) FROM test WHERE d + f > 3.0 AND d + f < 4.0;", dt);
    c("SELECT SUM(f + d) FROM test WHERE x - y = -35;", dt);
    c("SELECT SUM(f + d) FROM test WHERE x + y + 1 = 50;", dt);
    c("SELECT SUM(f * d + 15) FROM test WHERE x + y + 1 = 50;", dt);
    c("SELECT MIN(x), AVG(x * y), MAX(y + 7), AVG(x * f + 15), COUNT(*) FROM test WHERE x + y > 47 AND x + y < 51;",
      dt);
    c("SELECT AVG(f), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(y) > 42.0 ORDER BY n;", dt);
    c("SELECT AVG(f), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(f) > 1.09 ORDER BY n;", dt);
    c("SELECT AVG(f), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(f) > 1.09 AND AVG(y) > 42.0 ORDER BY n;",
      dt);
    c("SELECT AVG(d), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(d) > 2.2 AND AVG(y) > 42.0 ORDER BY n;",
      dt);
    c("SELECT AVG(f), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(d) > 2.2 AND AVG(y) > 42.0 ORDER BY n;",
      dt);
    c("SELECT AVG(f) + AVG(d), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(f) + AVG(d) > 3.0 ORDER BY n;",
      dt);
    c("SELECT AVG(f) + AVG(d), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(f) + AVG(d) > 3.5 ORDER BY n;",
      dt);
    c("SELECT f + d AS s, x * y FROM test ORDER by s DESC;", dt);
    c("SELECT COUNT(*) AS n FROM test GROUP BY f ORDER BY n;", dt);
    c("SELECT f, COUNT(*) FROM test GROUP BY f HAVING f > 1.25;", dt);
    c("SELECT COUNT(*) AS n FROM test GROUP BY d ORDER BY n;", dt);
    c("SELECT MIN(x + y) AS n FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY f + 1, f + d ORDER BY n;", dt);
    c("SELECT f + d AS s FROM test GROUP BY s ORDER BY s DESC;", dt);
    c("SELECT f + 1 AS s, AVG(u * f) FROM test GROUP BY s ORDER BY s DESC;", dt);
  }
}

TEST(Select, FilterAndMultipleAggregation) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT AVG(x), AVG(y) FROM test;", dt);
    c("SELECT MIN(x), AVG(x * y), MAX(y + 7), COUNT(*) FROM test WHERE x + y > 47 AND x + y < 51;", dt);
    c("SELECT str, AVG(x), COUNT(*) as xx, COUNT(*) as countval FROM test GROUP BY str ORDER BY str;", dt);
  }
}

TEST(Select, FilterAndGroupBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT MIN(x + y) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x, y;", dt);
    c("SELECT MIN(x + y) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x + 1, x + y;", dt);
    c("SELECT x, y, COUNT(*) FROM test GROUP BY x, y;", dt);
    c("SELECT x, dd, COUNT(*) FROM test GROUP BY x, dd ORDER BY x, dd;", dt);
    c("SELECT dd AS key1, COUNT(*) AS value1 FROM test GROUP BY key1 HAVING key1 IS NOT NULL ORDER BY key1, value1 "
      "DESC "
      "LIMIT 12;",
      dt);
    c("SELECT 'literal_string' AS key0 FROM test GROUP BY key0;", dt);
    c("SELECT str, MIN(y) FROM test WHERE y IS NOT NULL GROUP BY str ORDER BY str DESC;", dt);
    c("SELECT x, MAX(z) FROM test WHERE z IS NOT NULL GROUP BY x HAVING x > 7;", dt);
    c("SELECT CAST((dd - 0.5) * 2.0 AS int) AS key0, COUNT(*) AS val FROM test WHERE (dd >= 100.0 AND dd < 400.0) "
      "GROUP "
      "BY key0 HAVING key0 >= 0 AND key0 < 400 ORDER BY val DESC LIMIT 50 OFFSET 0;",
      dt);
    c("SELECT y, AVG(CASE WHEN x BETWEEN 6 AND 7 THEN x END) FROM test GROUP BY y ORDER BY y;", dt);
    c("SELECT x, AVG(u), COUNT(*) AS n FROM test GROUP BY x ORDER BY n DESC;", dt);
#ifdef HAVE_CALCITE
    c("SELECT x, COUNT(real_str) FROM test GROUP BY x ORDER BY x DESC;", dt);
    EXPECT_THROW(run_multiple_agg("SELECT x, MIN(real_str) FROM test GROUP BY x ORDER BY x DESC;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT x, MAX(real_str) FROM test GROUP BY x ORDER BY x DESC;", dt),
                 std::runtime_error);
#endif
  }
}

TEST(Select, FilterAndGroupByMultipleAgg) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT MIN(x + y), COUNT(*), AVG(x + 1) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x, y;", dt);
    c("SELECT MIN(x + y), COUNT(*), AVG(x + 1) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x + 1, x + y;", dt);
  }
}

TEST(Select, Having) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING MAX(x) > 5 ORDER BY n;", dt);
    c("SELECT MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING MAX(x) > 5 ORDER BY n LIMIT 1;", dt);
    c("SELECT MAX(y) AS n FROM test WHERE x > 7 GROUP BY z HAVING MAX(x) < 100 ORDER BY n;", dt);
    c("SELECT z, SUM(y) AS n FROM test WHERE x > 6 GROUP BY z HAVING MAX(x) < 100 ORDER BY n;", dt);
    c("SELECT z, SUM(y) AS n FROM test WHERE x > 6 GROUP BY z HAVING MAX(x) < 100 AND COUNT(*) > 5 ORDER BY n;", dt);
    c("SELECT z, SUM(y) AS n FROM test WHERE x > 6 GROUP BY z HAVING MAX(x) < 100 AND COUNT(*) > 9 ORDER BY n;", dt);
    c("SELECT str, COUNT(*) AS n FROM test GROUP BY str HAVING str IN ('bar', 'baz') ORDER BY str;", dt);
    c("SELECT str, COUNT(*) AS n FROM test GROUP BY str HAVING str LIKE 'ba_' ORDER BY str;", dt);
    c("SELECT ss, COUNT(*) AS n FROM test GROUP BY ss HAVING ss LIKE 'bo_' ORDER BY ss;", dt);
    c("SELECT x, COUNT(*) FROM test WHERE x > 9 GROUP BY x HAVING x > 15;", dt);
    c("SELECT x, AVG(y), AVG(y) FROM test GROUP BY x HAVING x >= 0 ORDER BY x;", dt);
    c("SELECT AVG(y), x, AVG(y) FROM test GROUP BY x HAVING x >= 0 ORDER BY x;", dt);
    c("SELECT x, y, COUNT(*) FROM test WHERE real_str LIKE 'nope%' GROUP BY x, y HAVING x >= 0 AND x < 12 AND y >= 0 "
      "AND y < 12 ORDER BY x, y;",
      dt);
  }
}

TEST(Select, CountDistinct) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(distinct x) FROM test;", dt);
    c("SELECT COUNT(distinct b) FROM test;", dt);
    c("SELECT COUNT(distinct f) FROM test;", dt);
    c("SELECT COUNT(distinct d) FROM test;", dt);
    c("SELECT COUNT(distinct str) FROM test;", dt);
    c("SELECT COUNT(distinct ss) FROM test;", dt);
    c("SELECT COUNT(distinct x + 1) FROM test;", dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x) FROM test GROUP BY y ORDER BY n;", dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x + 1) FROM test GROUP BY y ORDER BY n;",
      dt);
    c("SELECT COUNT(distinct dd) AS n FROM test GROUP BY y ORDER BY n;", dt);
    c("SELECT z, str, AVG(z), COUNT(distinct z) FROM test GROUP BY z, str;", dt);
    c("SELECT AVG(z), COUNT(distinct x) AS dx FROM test GROUP BY y HAVING dx > 1;", dt);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(distinct real_str) FROM test;", dt), std::runtime_error);
  }
}

TEST(Select, ScanNoAggregation) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT * FROM test;", dt);
    c("SELECT t.* FROM test t;", dt);
    c("SELECT x, z, t FROM test;", dt);
    c("SELECT x, y, x + 1 FROM test;", dt);
    c("SELECT x + z, t FROM test WHERE x <> 7 AND y > 42;", dt);
    c("SELECT * FROM test WHERE x > 8;", dt);
  }
}

TEST(Select, OrderBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    const auto rows = run_multiple_agg("SELECT x, y, z + t, x * y AS m FROM test ORDER BY 3 desc LIMIT 5;", dt);
    CHECK_EQ(rows.rowCount(), std::min(size_t(5), static_cast<size_t>(g_num_rows)));
    CHECK_EQ(rows.colCount(), size_t(4));
    for (size_t row_idx = 0; row_idx < rows.rowCount(); ++row_idx) {
      ASSERT_TRUE(v<int64_t>(rows.getRowAt(row_idx, 0, true)) == 8 || v<int64_t>(rows.getRowAt(row_idx, 0, true)) == 7);
      ASSERT_EQ(v<int64_t>(rows.getRowAt(row_idx, 1, true)), 43);
      ASSERT_EQ(v<int64_t>(rows.getRowAt(row_idx, 2, true)), 1104);
      ASSERT_TRUE(v<int64_t>(rows.getRowAt(row_idx, 3, true)) == 344 ||
                  v<int64_t>(rows.getRowAt(row_idx, 3, true)) == 301);
    }
    c("SELECT x, COUNT(distinct y) AS n FROM test GROUP BY x ORDER BY n DESC;", dt);
    c("SELECT x, x, COUNT(*) AS val FROM test GROUP BY x HAVING val > 5 ORDER BY val DESC LIMIT 5;", dt);
  }
}

TEST(Select, ComplexQueries) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) * MAX(y) - SUM(z) FROM test;", dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test WHERE z BETWEEN 100 AND 200 GROUP BY x, y;", dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test WHERE z BETWEEN 100 AND 200 "
      "GROUP BY x, y HAVING y > 2 * x AND MIN(y) > MAX(x);",
      dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test WHERE z BETWEEN 100 AND 200 "
      "GROUP BY x, y HAVING y > 2 * x AND MIN(y) > MAX(x) + 35;",
      dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test WHERE z BETWEEN 100 AND 200 "
      "GROUP BY x, y HAVING y > 2 * x AND MIN(y) > MAX(x) + 36;",
      dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test "
      "WHERE z BETWEEN 100 AND 200 GROUP BY a, y;",
      dt);
    const auto rows = run_multiple_agg(
        "SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test "
        "WHERE z BETWEEN 100 AND 200 GROUP BY x, y ORDER BY a DESC LIMIT 2;",
        dt);
    ASSERT_EQ(rows.rowCount(), size_t(2));
    {
      auto crt_row = rows.getNextRow(true, true);
      CHECK_EQ(size_t(2), crt_row.size());
      ASSERT_EQ(v<int64_t>(crt_row[0]), 51);
      ASSERT_EQ(v<int64_t>(crt_row[1]), -59 * g_num_rows / 2);
    }
    {
      auto crt_row = rows.getNextRow(true, true);
      CHECK_EQ(size_t(2), crt_row.size());
      ASSERT_EQ(v<int64_t>(crt_row[0]), 50);
      ASSERT_EQ(v<int64_t>(crt_row[1]), -59 * g_num_rows / 2);
    }
    auto empty_row = rows.getNextRow(true, true);
    CHECK(empty_row.empty());
  }
}

TEST(Select, GroupByExprNoFilterNoAggregate) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT x + y AS a FROM test GROUP BY a ORDER BY a;", dt);
#ifdef HAVE_RAVM
    ASSERT_EQ(8,
              v<int64_t>(run_simple_agg(
                  "SELECT TRUNCATE(x, 0) AS foo FROM test GROUP BY TRUNCATE(x, 0) ORDER BY foo DESC LIMIT 1;", dt)));
#endif  // HAVE_RAVM
  }
}

#ifdef HAVE_CALCITE
TEST(Select, DistinctProjection) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT DISTINCT str FROM test ORDER BY str;", dt);
#ifdef HAVE_RAVM
    c("SELECT DISTINCT(str), SUM(x) FROM test WHERE x > 7 GROUP BY str LIMIT 2;", dt);
#endif  // HAVE_RAVM
  }
}
#endif  // HAVE_CALCITE

TEST(Select, Case) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 WHEN x BETWEEN 8 AND 9 THEN 2 ELSE 3 END) FROM test;", dt);
    c("SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 END) FROM test;", dt);
    c("SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 WHEN x BETWEEN 8 AND 9 THEN 2 ELSE 3 END) "
      "FROM test WHERE CASE WHEN y BETWEEN 42 AND 43 THEN 5 ELSE 4 END > 4;",
      dt);
    ASSERT_EQ(std::numeric_limits<int64_t>::min(),
              v<int64_t>(run_simple_agg(
                  "SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 WHEN x BETWEEN 8 AND 9 THEN 2 ELSE 3 END) FROM test "
                  "WHERE CASE WHEN y BETWEEN 44 AND 45 THEN 5 ELSE 4 END > 4;",
                  dt)));
    c("SELECT CASE WHEN x + y > 50 THEN 77 ELSE 88 END AS foo, COUNT(*) FROM test GROUP BY foo ORDER BY foo;", dt);
    ASSERT_EQ(std::numeric_limits<double>::min(),
              v<double>(run_simple_agg(
                  "SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1.1 WHEN x BETWEEN 8 AND 9 THEN 2.2 ELSE 3.3 END) FROM "
                  "test WHERE CASE WHEN y BETWEEN 44 AND 45 THEN 5.1 ELSE 3.9 END > 4;",
                  dt)));
    c("SELECT CASE WHEN x BETWEEN 1 AND 3 THEN 'oops 1' WHEN x BETWEEN 4 AND 6 THEN 'oops 2' ELSE real_str END FROM "
      "test;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 3 THEN 'oops 1' WHEN x BETWEEN 4 AND 6 THEN 'oops 2' ELSE str END FROM test;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN 'eight' ELSE 'ooops' end FROM "
      "test;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN real_str ELSE 'ooops' END AS g "
      "FROM test;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN str ELSE 'ooops' END FROM test;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN 'eight' ELSE 'ooops' END FROM "
      "test;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN str WHEN x BETWEEN 7 AND 10 THEN 'eight' ELSE 'ooops' END AS g, "
      "COUNT(*) FROM test GROUP BY g ORDER BY g;",
      dt);
    c("SELECT y AS key0, SUM(CASE WHEN x > 7 THEN x / (x - 7) ELSE 99 END) FROM test GROUP BY key0 ORDER BY key0;", dt);
    c("SELECT CASE WHEN str IN ('str1', 'str3', 'str8') THEN 'foo' WHEN str IN ('str2', 'str4', 'str9') THEN 'bar' "
      "ELSE 'baz' END AS bucketed_str, COUNT(*) AS n FROM query_rewrite_test GROUP BY bucketed_str ORDER BY n DESC;",
      dt);
    c("SELECT CASE WHEN y > 40 THEN x END, x FROM test;", dt);
    c("SELECT COUNT(CASE WHEN str = 'foo' THEN 1 END) FROM test;", dt);
    c("SELECT COUNT(CASE WHEN str = 'foo' THEN 1 ELSE NULL END) FROM test;", dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 3 THEN y ELSE y END AS foobar FROM test ORDER BY foobar DESC;", dt);
    c("SELECT x, AVG(CASE WHEN y BETWEEN 41 AND 42 THEN y END) FROM test GROUP BY x ORDER BY x;", dt);
    c("SELECT x, SUM(CASE WHEN y BETWEEN 41 AND 42 THEN y END) FROM test GROUP BY x ORDER BY x;", dt);
    c("SELECT x, COUNT(CASE WHEN y BETWEEN 41 AND 42 THEN y END) FROM test GROUP BY x ORDER BY x;", dt);
    c("SELECT CASE WHEN x > 8 THEN 'oops' ELSE 'ok' END FROM test LIMIT 1;", dt);
    c("SELECT CASE WHEN x < 9 THEN 'ok' ELSE 'oops' END FROM test LIMIT 1;", dt);
    c("SELECT CASE WHEN str IN ('foo', 'bar') THEN str END key1, COUNT(*) FROM test GROUP BY str HAVING key1 IS NOT "
      "NULL ORDER BY key1;",
      dt);
#ifdef HAVE_CALCITE
    ASSERT_EQ(
        int64_t(1),
        v<int64_t>(run_simple_agg("SELECT MIN(CASE WHEN x BETWEEN 7 AND 8 THEN true ELSE false END) FROM test;", dt)));
    ASSERT_EQ(
        int64_t(0),
        v<int64_t>(run_simple_agg("SELECT MIN(CASE WHEN x BETWEEN 6 AND 7 THEN true ELSE false END) FROM test;", dt)));
#else
    ASSERT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "SELECT MIN(CASE WHEN x BETWEEN 7 AND 8 THEN CAST('t' AS boolean) ELSE CAST('f' AS boolean) END) "
                  "FROM test;",
                  dt)));
    ASSERT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg(
                  "SELECT MIN(CASE WHEN x BETWEEN 6 AND 7 THEN CAST('t' AS boolean) ELSE CAST('f' AS boolean) END) "
                  "FROM test;",
                  dt)));
#endif  // HAVE_CALCITE
  }
}

TEST(Select, Strings) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT str, COUNT(*) FROM test GROUP BY str HAVING COUNT(*) > 5 ORDER BY str;", dt);
    c("SELECT str, COUNT(*) FROM test WHERE str = 'bar' GROUP BY str HAVING COUNT(*) > 4 ORDER BY str;", dt);
    c("SELECT str, COUNT(*) FROM test WHERE str = 'bar' GROUP BY str HAVING COUNT(*) > 5 ORDER BY str;", dt);
    c("SELECT str, COUNT(*) FROM test GROUP BY str ORDER BY str;", dt);
    c("SELECT COUNT(*) FROM test WHERE str IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE str IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE ss IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE ss IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE '%%%';", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE 'ba%';", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE '%eal_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE '%ba%';", dt);
    c("SELECT * FROM test WHERE str LIKE '%';", dt);
    c("SELECT * FROM test WHERE str LIKE 'f%%';", dt);
    c("SELECT * FROM test WHERE str LIKE 'f%\%';", dt);
    c("SELECT * FROM test WHERE ss LIKE 'f%\%';", dt);
    c("SELECT * FROM test WHERE str LIKE '@f%%' ESCAPE '@';", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE 'ba_' or str LIKE 'fo_';", dt);
    c("SELECT COUNT(*) FROM test WHERE str IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE str IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE str > 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE str > 'fo';", dt);
    c("SELECT COUNT(*) FROM test WHERE str >= 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' < str;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'fo' < str;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' <= str;", dt);
    c("SELECT COUNT(*) FROM test WHERE str = 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' = str;", dt);
    c("SELECT COUNT(*) FROM test WHERE str <> 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' <> str;", dt);
    c("SELECT COUNT(*) FROM test WHERE str = 'foo' OR str = 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE str = real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE str <> str;", dt);
    c("SELECT COUNT(*) FROM test WHERE LENGTH(str) = 3;", dt);
    c("SELECT fixed_str, COUNT(*) FROM test GROUP BY fixed_str HAVING COUNT(*) > 5 ORDER BY fixed_str;", dt);
    c("SELECT fixed_str, COUNT(*) FROM test WHERE fixed_str = 'bar' GROUP BY fixed_str HAVING COUNT(*) > 4 ORDER BY "
      "fixed_str;",
      dt);
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CHAR_LENGTH(str) = 3;", dt)));
    ASSERT_EQ(g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str ILIKE 'f%%';", dt)));
    ASSERT_EQ(0, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str ILIKE 'McDonald''s';", dt)));
#ifdef HAVE_CALCITE
    ASSERT_EQ("foo",
              boost::get<std::string>(
                  v<NullableString>(run_simple_agg("SELECT str FROM test WHERE REGEXP_LIKE(str, '^f.?.+');", dt))));
    ASSERT_EQ("bar",
              boost::get<std::string>(
                  v<NullableString>(run_simple_agg("SELECT str FROM test WHERE REGEXP_LIKE(str, '^[a-z]+r$');", dt))));
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP '.*';", dt)));
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP '...';", dt)));
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP '.+.+.+';", dt)));
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP '.?.?.?';", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP 'ba.' or str REGEXP 'fo.';", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(
                  run_simple_agg("SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(str, 'ba.') or str REGEXP 'fo.?';", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(
                  run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP 'ba.' or REGEXP_LIKE(str, 'fo.+');", dt)));
    ASSERT_EQ(g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP 'ba.+';", dt)));
    ASSERT_EQ(g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(str, '.?ba.*');", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(str,'ba.') or REGEXP_LIKE(str, 'fo.+');", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(
                  run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP 'ba.' or REGEXP_LIKE(str, 'fo.+');", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(
                  run_simple_agg("SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(str, 'ba.') or str REGEXP 'fo.?';", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP 'ba.' or str REGEXP 'fo.';", dt)));
#endif  // HAVE_CALCITE
  }
}

TEST(Select, StringsNoneEncoding) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test WHERE real_str LIKE 'real_%%%';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str LIKE 'real_ba%';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str LIKE '%eal_bar';", dt);
    c("SELECT * FROM test WHERE real_str LIKE '%';", dt);
    c("SELECT * FROM test WHERE real_str LIKE 'real_f%%';", dt);
    c("SELECT * FROM test WHERE real_str LIKE 'real_f%\%';", dt);
    c("SELECT * FROM test WHERE real_str LIKE 'real_@f%%' ESCAPE '@';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str LIKE 'real_ba_' or real_str LIKE 'real_fo_';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str > 'real_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str > 'real_fo';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str >= 'real_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'real_bar' < real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'real_fo' < real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'real_bar' <= real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str = 'real_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'real_bar' = real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str <> 'real_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'real_bar' <> real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str = 'real_foo' OR real_str = 'real_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str = real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str <> real_str;", dt);
    ASSERT_EQ(g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE real_str ILIKE 'rEaL_f%%';", dt)));
    c("SELECT COUNT(*) FROM test WHERE LENGTH(real_str) = 8;", dt);
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CHAR_LENGTH(real_str) = 8;", dt)));
#ifdef HAVE_CALCITE
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(real_str,'real_.*.*.*');", dt)));
    ASSERT_EQ(g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE real_str REGEXP 'real_ba.*';", dt)));
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE real_str REGEXP '.*';", dt)));
    ASSERT_EQ(g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE real_str REGEXP 'real_f.*.*';", dt)));
    ASSERT_EQ(0, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE real_str REGEXP 'real_f.+\%';", dt)));
#endif  // HAVE_CALCITE
  }
}

namespace {

void check_date_trunc_groups(const ResultRows& rows) {
  {
    const auto crt_row = rows.getNextRow(true, true);
    CHECK(!crt_row.empty());
    CHECK_EQ(size_t(3), crt_row.size());
    const auto sv0 = v<int64_t>(crt_row[0]);
    ASSERT_EQ(int64_t(936144000), sv0);
    const auto sv1 = boost::get<std::string>(v<NullableString>(crt_row[1]));
    ASSERT_EQ("foo", sv1);
    const auto sv2 = v<int64_t>(crt_row[2]);
    ASSERT_EQ(g_num_rows, sv2);
  }
  {
    const auto crt_row = rows.getNextRow(true, true);
    CHECK(!crt_row.empty());
    CHECK_EQ(size_t(3), crt_row.size());
    const auto sv0 = v<int64_t>(crt_row[0]);
    ASSERT_EQ(inline_int_null_val(rows.getColType(0)), sv0);
    const auto sv1 = boost::get<std::string>(v<NullableString>(crt_row[1]));
    ASSERT_EQ("bar", sv1);
    const auto sv2 = v<int64_t>(crt_row[2]);
    ASSERT_EQ(g_num_rows / 2, sv2);
  }
  {
    const auto crt_row = rows.getNextRow(true, true);
    CHECK(!crt_row.empty());
    CHECK_EQ(size_t(3), crt_row.size());
    const auto sv0 = v<int64_t>(crt_row[0]);
    ASSERT_EQ(int64_t(936144000), sv0);
    const auto sv1 = boost::get<std::string>(v<NullableString>(crt_row[1]));
    ASSERT_EQ("baz", sv1);
    const auto sv2 = v<int64_t>(crt_row[2]);
    ASSERT_EQ(g_num_rows / 2, sv2);
  }
  const auto crt_row = rows.getNextRow(true, true);
  CHECK(crt_row.empty());
}

}  // namespace

TEST(Select, Time) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST('1999-09-10' AS DATE) > o;", dt)));
    ASSERT_EQ(0, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST('1999-09-10' AS DATE) <= o;", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST('15:13:15' AS TIME) > n;", dt)));
    ASSERT_EQ(0, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST('15:13:15' AS TIME) <= n;", dt)));
    cta("SELECT DATETIME('NOW') FROM test limit 1;", dt);
    // these next tests work because all dates are before now 2015-12-8 17:00:00
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m < NOW();", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m > timestamp(0) '2014-12-13T000000';", dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST(o AS TIMESTAMP) > timestamp(0) '1999-09-08T160000';", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST(o AS TIMESTAMP) > timestamp(0) '1999-09-10T160000';", dt)));
    ASSERT_EQ(14185093950L, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(EPOCH FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(20140, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(YEAR FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(120, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MONTH FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(130, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DAY FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(22, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(HOUR FROM m)) FROM test;", dt)));
    ASSERT_EQ(23, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MINUTE FROM m)) FROM test;", dt)));
    ASSERT_EQ(15, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(SECOND FROM m)) FROM test;", dt)));
    ASSERT_EQ(6, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DOW FROM m)) FROM test;", dt)));
    ASSERT_EQ(347, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DOY FROM m)) FROM test;", dt)));
    ASSERT_EQ(15, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(HOUR FROM n)) FROM test;", dt)));
    ASSERT_EQ(13, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MINUTE FROM n)) FROM test;", dt)));
    ASSERT_EQ(14, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(SECOND FROM n)) FROM test;", dt)));
    ASSERT_EQ(1999, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(YEAR FROM o)) FROM test;", dt)));
    ASSERT_EQ(9, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MONTH FROM o)) FROM test;", dt)));
    ASSERT_EQ(9, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DAY FROM o)) FROM test;", dt)));
    ASSERT_EQ(4, v<int64_t>(run_simple_agg("SELECT EXTRACT(DOW FROM o) FROM test;", dt)));
    ASSERT_EQ(252, v<int64_t>(run_simple_agg("SELECT EXTRACT(DOY FROM o) FROM test;", dt)));
    ASSERT_EQ(936835200L, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(EPOCH FROM o)) FROM test;", dt)));
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg(
                  "SELECT MAX(EXTRACT(WEEK FROM CAST('2012-01-01 20:15:12' AS TIMESTAMP))) FROM test limit 1;", dt)));
    ASSERT_EQ(10L,
              v<int64_t>(run_simple_agg(
                  "SELECT MAX(EXTRACT(WEEK FROM CAST('2008-03-03 20:15:12' AS TIMESTAMP))) FROM test limit 1;", dt)));
    // Monday
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(DOW FROM CAST('2008-03-03 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    // Monday
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(ISODOW FROM CAST('2008-03-03 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    // Sunday
    ASSERT_EQ(0L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(DOW FROM CAST('2008-03-02 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    // Sunday
    ASSERT_EQ(7L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(ISODOW FROM CAST('2008-03-02 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));

    // do some DATE_TRUNC tests
    /*
 * year
 * month
 * day
 * hour
 * minute
 * second
 *
 * millennium
 * century
 * decade
 * milliseconds
 * microseconds
 * week
 */
    ASSERT_EQ(1325376000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(year, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(1335830400L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(month, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(1336435200L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(day, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(1336507200L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(hour, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(1336508112L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(second, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(978307200L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(millennium, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(978307200L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(century, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(1293840000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(decade, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(1336508112L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(millisecond, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(1336508112L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(microsecond, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(1336262400L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(week, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));

    ASSERT_EQ(-2114380800L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(year, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2104012800L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(month, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2103408000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(day, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2103336000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(hour, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2103335088L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(second, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-30578688000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(millennium, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2177452800L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(century, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2177452800L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(decade, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2103335088L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(millisecond, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2103335088L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(microsecond, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2103840000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(week, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));

    ASSERT_EQ(31536000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(decade, CAST('1972-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(662688000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(decade, CAST('2000-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    // test QUARTER
    ASSERT_EQ(4,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT(quarter FROM CAST('2008-11-27 12:12:12' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT(quarter FROM CAST('2008-03-21 12:12:12' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1199145600L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC(quarter, CAST('2008-03-21 12:12:12' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1230768000L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC(quarter, CAST('2009-03-21 12:12:12' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1254355200L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC(quarter, CAST('2009-11-21 12:12:12' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(946684800L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC(quarter, CAST('2000-03-21 12:12:12' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2208988800L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC(quarter, CAST('1900-03-21 12:12:12' AS timestamp)) FROM test limit 1;", dt)));
    // test DATE format processing
    ASSERT_EQ(1434844800L, v<int64_t>(run_simple_agg("select CAST('2015-06-21' AS DATE) FROM test limit 1;", dt)));
    ASSERT_EQ(1434844800L, v<int64_t>(run_simple_agg("select CAST('06/21/2015' AS DATE) FROM test limit 1;", dt)));
    ASSERT_EQ(1434844800L, v<int64_t>(run_simple_agg("select CAST('21-Jun-15' AS DATE) FROM test limit 1;", dt)));
    ASSERT_EQ(1434844800L, v<int64_t>(run_simple_agg("select CAST('21/Jun/2015' AS DATE) FROM test limit 1;", dt)));
    ASSERT_EQ(1434844800L, v<int64_t>(run_simple_agg("select CAST('1434844800' AS DATE) FROM test limit 1;", dt)));

    // test different input formats
    // added new format for verizon
    ASSERT_EQ(1434896116L,
              v<int64_t>(run_simple_agg("select CAST('2015-06-21 14:15:16' AS timestamp) FROM test limit 1;", dt)));
    ASSERT_EQ(
        1434896116L,
        v<int64_t>(run_simple_agg("select CAST('21-JUN-15 2.15.16.12345 PM' AS timestamp) FROM test limit 1;", dt)));
    ASSERT_EQ(
        1434852916L,
        v<int64_t>(run_simple_agg("select CAST('21-JUN-15 2.15.16.12345 AM' AS timestamp) FROM test limit 1;", dt)));
    ASSERT_EQ(1434896116L,
              v<int64_t>(run_simple_agg("select CAST('06/21/2015 14:15:16' AS timestamp) FROM test limit 1;", dt)));

    // Support ISO date offset format
    ASSERT_EQ(
        1440180794L,
        v<int64_t>(run_simple_agg("select CAST('21/Aug/2015:12:13:14 -0600' AS timestamp) FROM test limit 1;", dt)));
    ASSERT_EQ(
        1440180794L,
        v<int64_t>(run_simple_agg("select CAST('2015-08-21T12:13:14 -0600' AS timestamp) FROM test limit 1;", dt)));
    ASSERT_EQ(
        1440180794L,
        v<int64_t>(run_simple_agg("select CAST('21-Aug-15 12:13:14 -0600' AS timestamp) FROM test limit 1;", dt)));
    ASSERT_EQ(
        1440180794L,
        v<int64_t>(run_simple_agg("select CAST('21/Aug/2015:13:13:14 -0500' AS timestamp) FROM test limit 1;", dt)));
    ASSERT_EQ(1440180794L,
              v<int64_t>(run_simple_agg("select CAST('2015-08-21T18:13:14' AS timestamp) FROM test limit 1;", dt)));
    // add test for quarterday behaviour
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T04:23:11' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T00:00:00' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(2L,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T08:23:11' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(3L,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T14:23:11' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(4L,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T23:23:11' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1440115200L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC (QUARTERDAY, CAST('2015-08-21T04:23:11' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1440136800L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC (QUARTERDAY, CAST('2015-08-21T08:23:11' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1440158400L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC (QUARTERDAY, CAST('2015-08-21T13:23:11' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1440180000L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC (QUARTERDAY, CAST('2015-08-21T23:59:59' AS timestamp)) FROM test limit 1;", dt)));
#ifdef HAVE_CALCITE
    ASSERT_EQ(
        2007,
        v<int64_t>(run_simple_agg("SELECT DATEPART('year', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        2007,
        v<int64_t>(run_simple_agg("SELECT DATEPART('yyyy', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        2007,
        v<int64_t>(run_simple_agg("SELECT DATEPART('yy', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(4,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('quarter', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        4,
        v<int64_t>(run_simple_agg("SELECT DATEPART('qq', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        4, v<int64_t>(run_simple_agg("SELECT DATEPART('q', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(10,
              v<int64_t>(
                  run_simple_agg("SELECT DATEPART('month', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        10,
        v<int64_t>(run_simple_agg("SELECT DATEPART('mm', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        10,
        v<int64_t>(run_simple_agg("SELECT DATEPART('m', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(303,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('dayofyear', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        303,
        v<int64_t>(run_simple_agg("SELECT DATEPART('dy', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        303,
        v<int64_t>(run_simple_agg("SELECT DATEPART('y', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        30,
        v<int64_t>(run_simple_agg("SELECT DATEPART('day', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        30,
        v<int64_t>(run_simple_agg("SELECT DATEPART('dd', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        30,
        v<int64_t>(run_simple_agg("SELECT DATEPART('d', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        12,
        v<int64_t>(run_simple_agg("SELECT DATEPART('hour', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        12,
        v<int64_t>(run_simple_agg("SELECT DATEPART('hh', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('minute', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("SELECT DATEPART('mi', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("SELECT DATEPART('n', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(32,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('second', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        32,
        v<int64_t>(run_simple_agg("SELECT DATEPART('ss', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        32,
        v<int64_t>(run_simple_agg("SELECT DATEPART('s', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(3,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('year', CAST('2006-01-07 00:00:00' as TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                  "TIMESTAMP)) FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(36,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('month', CAST('2006-01-07 00:00:00' as TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                  "TIMESTAMP)) FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1095,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('day', CAST('2006-01-07 00:00:00' as TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                  "TIMESTAMP)) FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(12,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('quarter', CAST('2006-01-07 00:00:00' as TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                  "TIMESTAMP)) FROM TEST LIMIT 1;",
                  dt)));
#endif  // HAVE_CALCITE
    ASSERT_EQ(1418428800L, v<int64_t>(run_simple_agg("SELECT CAST(m AS date) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(1336435200L,
              v<int64_t>(run_simple_agg(
                  "SELECT CAST(CAST('2012-05-08 20:15:12' AS TIMESTAMP) AS DATE) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(g_num_rows * 2, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test GROUP BY CAST(m AS date);", dt)));
    EXPECT_THROW(run_simple_agg("SELECT DATEPART('s', TIMESTAMP '2007-10-30 12:15:32') FROM test;", dt),
                 std::runtime_error);
    const auto rows = run_multiple_agg(
        "SELECT DATE_TRUNC(month, CAST(o AS TIMESTAMP(0))) AS key0, str AS key1, COUNT(*) AS val FROM test GROUP BY "
        "key0, key1 ORDER BY val DESC, key1;",
        dt);
    check_date_trunc_groups(rows);
  }
}

TEST(Select, In) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test WHERE x IN (7, 8);", dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (9, 10);", dt);
    c("SELECT COUNT(*) FROM test WHERE z IN (101, 102);", dt);
    c("SELECT COUNT(*) FROM test WHERE z IN (201, 202);", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str IN ('real_foo', 'real_bar');", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str IN ('real_foo', 'real_bar', 'real_baz', 'foo');", dt);
    c("SELECT COUNT(*) FROM test WHERE str IN ('foo', 'bar', 'real_foo');", dt);
  }
}

TEST(Select, DivByZero) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT x / 0 FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(distinct x / 0) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT f / 0. FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT d / 0. FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT f / (f - f) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test GROUP BY y / (x - x);", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test GROUP BY z, y / (x - x);", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test GROUP BY MOD(y , (x - x));", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT SUM(x) / SUM(CASE WHEN str = 'none' THEN y ELSE 0 END) FROM test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("SELECT COUNT(*) FROM test WHERE y / (x - x) = 0;", dt), std::runtime_error);
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x = x OR  y / (x - x) = y;", dt)));
  }
}

TEST(Select, OverflowAndUnderFlow) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test WHERE z + 32600 > 0;", dt);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE z + 32666 > 0;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE -32670 - z < 0;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE (z + 16333) * 2 > 0;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE x + 2147483640 > 0;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE -x - 2147483642 < 0;", dt), std::runtime_error);
    c("SELECT COUNT(*) FROM test WHERE t + 9223372036854774000 > 0;", dt);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE t + 9223372036854775000 > 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE -t - 9223372036854775000 < 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE ofd + x - 2 > 0;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE ufd * 3 - ofd * 1024 < -2;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE ofd * 2 > 0;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE ofq + 1 > 0;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE -ufq - 9223372036854775000 > 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE -92233720368547758 - ofq <= 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT cast((z - -32666) *0.000190 as int) as key0, "
                     "COUNT(*) AS val FROM test WHERE (z >= -32666 AND z < 31496) "
                     "GROUP BY key0 HAVING key0 >= 0 AND key0 < 12 ORDER BY val "
                     "DESC LIMIT 50 OFFSET 0;",
                     dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT dd * 2000000000000000 FROM test LIMIT 5;", dt), std::runtime_error);
    c("SELECT dd * 200000000000000 FROM test LIMIT 5;", dt);  // overflow avoided through decimal mul optimization
    c("SELECT cast((cast(z as int) - -32666) *0.000190 as int) as key0, "
      "COUNT(*) AS val FROM test WHERE (z >= -32666 AND z < 31496) "
      "GROUP BY key0 HAVING key0 >= 0 AND key0 < 12 ORDER BY val "
      "DESC LIMIT 50 OFFSET 0;",
      dt);
#ifdef ENABLE_COMPACTION
    c("SELECT SUM(ofd) FROM test GROUP BY x;", dt);
    c("SELECT SUM(ufd) FROM test GROUP BY x;", dt);
    EXPECT_THROW(run_multiple_agg("SELECT SUM(ofq) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT SUM(ufq) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT SUM(ofq) FROM test GROUP BY x;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT SUM(ufq) FROM test GROUP BY x;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT AVG(ofq) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT AVG(ufq) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT AVG(ofq) FROM test GROUP BY y;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT AVG(ufq) FROM test GROUP BY y;", dt), std::runtime_error);
#endif  // ENABLE_COMPACTION
  }
}

TEST(Select, BooleanColumn) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(g_num_rows + g_num_rows / 2, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE b;", dt)));
    ASSERT_EQ(g_num_rows / 2, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE NOT b;", dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x < 8 AND b;", dt)));
    ASSERT_EQ(0, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x < 8 AND NOT b;", dt)));
    ASSERT_EQ(7, v<int64_t>(run_simple_agg("SELECT MAX(x) FROM test WHERE b = CAST('t' AS boolean);", dt)));
    ASSERT_EQ(3 * g_num_rows,
              v<int64_t>(run_simple_agg(" SELECT SUM(2 *(CASE when x = 7 then 1 else 0 END)) FROM test;", dt)));
    c("SELECT COUNT(*) AS n FROM test GROUP BY x = 7, b ORDER BY n;", dt);
  }
}

TEST(Select, UnsupportedCast) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT CAST(x AS VARCHAR) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(f AS VARCHAR) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(d AS VARCHAR) FROM test;", dt), std::runtime_error);
  }
}

TEST(Select, CastFromLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT CAST(2.3 AS SMALLINT) FROM test;", dt);
    c("SELECT CAST(2.3 AS INT) FROM test;", dt);
    c("SELECT CAST(2.3 AS BIGINT) FROM test;", dt);
    c("SELECT CAST(2.3 AS FLOAT) FROM test;", dt);
    c("SELECT CAST(2.3 AS DOUBLE) FROM test;", dt);
    c("SELECT CAST(2.3 AS DECIMAL(2, 1)) FROM test;", dt);
    c("SELECT CAST(2.3 AS NUMERIC(2, 1)) FROM test;", dt);
  }
}

#ifdef HAVE_CALCITE
TEST(Select, CastFromNull) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT CAST(NULL AS SMALLINT) FROM test;", dt);
    c("SELECT CAST(NULL AS INT) FROM test;", dt);
    c("SELECT CAST(NULL AS BIGINT) FROM test;", dt);
    c("SELECT CAST(NULL AS FLOAT) FROM test;", dt);
    c("SELECT CAST(NULL AS DOUBLE) FROM test;", dt);
    c("SELECT CAST(NULL AS DECIMAL) FROM test;", dt);
    c("SELECT CAST(NULL AS NUMERIC) FROM test;", dt);
  }
}
#endif  // HAVE_CALCITE

#ifdef HAVE_CALCITE
TEST(Select, TimeInterval) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(60 * 60 * 1000L, v<int64_t>(run_simple_agg("SELECT INTERVAL '1' HOUR FROM test LIMIT 1;", dt)));
    ASSERT_EQ(24 * 60 * 60 * 1000L, v<int64_t>(run_simple_agg("SELECT INTERVAL '1' DAY FROM test LIMIT 1;", dt)));
    ASSERT_EQ(
        2 * g_num_rows,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE INTERVAL '1' MONTH < INTERVAL '2' MONTH;", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE INTERVAL '1' DAY < INTERVAL '2' DAY;", dt)));
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test GROUP BY INTERVAL '1' DAY;", dt)));
    ASSERT_EQ(3 * 60 * 60 * 1000L, v<int64_t>(run_simple_agg("SELECT 3 * INTERVAL '1' HOUR FROM test LIMIT 1;", dt)));
    ASSERT_EQ(3 * 60 * 60 * 1000L, v<int64_t>(run_simple_agg("SELECT INTERVAL '1' HOUR * 3 FROM test LIMIT 1;", dt)));
    ASSERT_EQ(7L, v<int64_t>(run_simple_agg("SELECT INTERVAL '1' MONTH * x FROM test WHERE x <> 8 LIMIT 1;", dt)));
    ASSERT_EQ(7L, v<int64_t>(run_simple_agg("SELECT x * INTERVAL '1' MONTH FROM test WHERE x <> 8 LIMIT 1;", dt)));
    ASSERT_EQ(42L, v<int64_t>(run_simple_agg("SELECT INTERVAL '1' MONTH * y FROM test WHERE y <> 43 LIMIT 1;", dt)));
    ASSERT_EQ(42L, v<int64_t>(run_simple_agg("SELECT y * INTERVAL '1' MONTH FROM test WHERE y <> 43 LIMIT 1;", dt)));
    ASSERT_EQ(1002L,
              v<int64_t>(run_simple_agg("SELECT INTERVAL '1' MONTH * t FROM test WHERE t <> 1001 LIMIT 1;", dt)));
    ASSERT_EQ(1002L,
              v<int64_t>(run_simple_agg("SELECT t * INTERVAL '1' MONTH FROM test WHERE t <> 1001 LIMIT 1;", dt)));
    ASSERT_EQ(3L, v<int64_t>(run_simple_agg("SELECT INTERVAL '1' MONTH + INTERVAL '2' MONTH FROM test LIMIT 1;", dt)));
#ifdef HAVE_RAVM
    ASSERT_EQ(1388534400L,
              v<int64_t>(run_simple_agg(
                  "SELECT CAST(m AS date) + CAST(TRUNCATE(-1 * (EXTRACT(DOY FROM m) - 1), 0) AS INTEGER) * INTERVAL "
                  "'1' DAY AS g FROM test GROUP BY g;",
                  dt)));
    ASSERT_EQ(1417392000L,
              v<int64_t>(run_simple_agg(
                  "SELECT CAST(m AS date) + CAST(TRUNCATE(-1 * (EXTRACT(DAY FROM m) - 1), 0) AS INTEGER) * INTERVAL "
                  "'1' DAY AS g FROM test GROUP BY g;",
                  dt)));
    ASSERT_EQ(1418508000L,
              v<int64_t>(run_simple_agg(
                  "SELECT CAST(m AS date) + EXTRACT(HOUR FROM m) * INTERVAL '1' HOUR AS g FROM test GROUP BY g;", dt)));
    ASSERT_EQ(1388534400L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(SQL_TSI_DAY, CAST(TRUNCATE(-1 * (EXTRACT(DOY from m) - 1), 0) AS INTEGER), "
                  "CAST(m AS DATE)) AS g FROM test GROUP BY g;",
                  dt)));
    ASSERT_EQ(1417392000L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(SQL_TSI_DAY, CAST(TRUNCATE(-1 * (EXTRACT(DAY from m) - 1), 0) AS INTEGER), "
                  "CAST(m AS DATE)) AS g FROM test GROUP BY g;",
                  dt)));
    ASSERT_EQ(1418508000L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(SQL_TSI_HOUR, EXTRACT(HOUR from m), CAST(m AS DATE)) AS g FROM test GROUP BY g;",
                  dt)));
#endif  // HAVE_RAVM
  }
}
#endif  // HAVE_CALCITE

TEST(Select, UnsupportedNodes) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT 1 + 2;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT *;", dt), std::runtime_error);
  }
}

namespace Importer_NS {

ArrayDatum StringToArray(const std::string& s, const SQLTypeInfo& ti, const CopyParams& copy_params);
void parseStringArray(const std::string& s, const CopyParams& copy_params, std::vector<std::string>& string_vec);

}  // Importer_NS

namespace {

const size_t g_array_test_row_count{20};

void import_array_test(const std::string& table_name) {
  CHECK_EQ(size_t(0), g_array_test_row_count % 4);
  const auto& cat = g_session->get_catalog();
  const auto td = cat.getMetadataForTable(table_name);
  CHECK(td);
  Importer_NS::Loader loader(cat, td);
  std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>> import_buffers;
  const auto col_descs = cat.getAllColumnMetadataForTable(td->tableId, false, false);
  for (const auto cd : col_descs) {
    import_buffers.emplace_back(new Importer_NS::TypedImportBuffer(
        cd,
        cd->columnType.get_compression() == kENCODING_DICT
            ? cat.getMetadataForDict(cd->columnType.get_comp_param())->stringDict.get()
            : nullptr));
  }
  Importer_NS::CopyParams copy_params;
  copy_params.array_begin = '{';
  copy_params.array_end = '}';
  for (size_t row_idx = 0; row_idx < g_array_test_row_count; ++row_idx) {
    for (const auto& import_buffer : import_buffers) {
      const auto& ti = import_buffer->getTypeInfo();
      switch (ti.get_type()) {
        case kINT:
          import_buffer->addInt(7 + row_idx);
          break;
        case kARRAY: {
          const auto& elem_ti = ti.get_elem_type();
          std::vector<std::string> array_elems;
          switch (elem_ti.get_type()) {
            case kBOOLEAN: {
              for (size_t i = 0; i < 3; ++i) {
                if (row_idx % 2) {
                  array_elems.push_back("T");
                  array_elems.push_back("F");
                } else {
                  array_elems.push_back("F");
                  array_elems.push_back("T");
                }
              }
              break;
            }
            case kSMALLINT:
              for (size_t i = 0; i < 3; ++i) {
                array_elems.push_back(std::to_string(row_idx + i + 1));
              }
              break;
            case kINT:
              for (size_t i = 0; i < 3; ++i) {
                array_elems.push_back(std::to_string((row_idx + i + 1) * 10));
              }
              break;
            case kBIGINT:
              for (size_t i = 0; i < 3; ++i) {
                array_elems.push_back(std::to_string((row_idx + i + 1) * 100));
              }
              break;
            case kTEXT:
              for (size_t i = 0; i < 3; ++i) {
                array_elems.emplace_back(2, 'a' + row_idx + i);
              }
              break;
            case kFLOAT:
              for (size_t i = 0; i < 3; ++i) {
                array_elems.emplace_back(std::to_string(row_idx + i + 1) + "." + std::to_string(row_idx + i + 1));
              }
              break;
            case kDOUBLE:
              for (size_t i = 0; i < 3; ++i) {
                array_elems.emplace_back(std::to_string(11 * (row_idx + i + 1)) + "." +
                                         std::to_string(row_idx + i + 1));
              }
              break;
            default:
              CHECK(false);
          }
          if (elem_ti.is_string()) {
            import_buffer->addDictEncodedStringArray({array_elems});
          } else {
            auto arr_str = "{" + boost::algorithm::join(array_elems, ",") + "}";
            import_buffer->addArray(StringToArray(arr_str, ti, copy_params));
          }
          break;
        }
        case kTEXT:
          import_buffer->addString("real_str" + std::to_string(row_idx));
          break;
        default:
          CHECK(false);
      }
    }
  }
  loader.load(import_buffers, g_array_test_row_count);
  loader.checkpoint();
}

void import_gpu_sort_test() {
  const std::string drop_old_gpu_sort_test{"DROP TABLE IF EXISTS gpu_sort_test;"};
  run_ddl_statement(drop_old_gpu_sort_test);
  g_sqlite_comparator.query(drop_old_gpu_sort_test);
  const std::string create_gpu_sort_test{"CREATE TABLE gpu_sort_test(x int);"};
  run_ddl_statement("CREATE TABLE gpu_sort_test(x int) WITH (fragment_size=2);");
  g_sqlite_comparator.query("CREATE TABLE gpu_sort_test(x int);");
  for (size_t i = 0; i < 4; ++i) {
    const std::string insert_query{"INSERT INTO gpu_sort_test VALUES(2);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (size_t i = 0; i < 6; ++i) {
    const std::string insert_query{"INSERT INTO gpu_sort_test VALUES(16000);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
}

void import_query_rewrite_test() {
  const std::string drop_old_query_rewrite_test{"DROP TABLE IF EXISTS query_rewrite_test;"};
  run_ddl_statement(drop_old_query_rewrite_test);
  g_sqlite_comparator.query(drop_old_query_rewrite_test);
  run_ddl_statement("CREATE TABLE query_rewrite_test(x int, str text encoding dict) WITH (fragment_size=2);");
  g_sqlite_comparator.query("CREATE TABLE query_rewrite_test(x int, str text);");
  for (size_t i = 1; i <= 30; ++i) {
    for (size_t j = 1; j <= i % 2 + 1; ++j) {
      const std::string insert_query{"INSERT INTO query_rewrite_test VALUES(" + std::to_string(i) + ", 'str" +
                                     std::to_string(i) + "');"};
      run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
      g_sqlite_comparator.query(insert_query);
    }
  }
}

void import_big_decimal_range_test() {
  const std::string drop_old_decimal_range_test("DROP TABLE IF EXISTS big_decimal_range_test;");
  run_ddl_statement(drop_old_decimal_range_test);
  g_sqlite_comparator.query(drop_old_decimal_range_test);
  run_ddl_statement("CREATE TABLE big_decimal_range_test(d DECIMAL(14, 2), d1 DECIMAL(17,11)) WITH (fragment_size=2);");
  g_sqlite_comparator.query("CREATE TABLE big_decimal_range_test(d DECIMAL(14, 2), d1 DECIMAL(17,11));");
  {
    const std::string insert_query{"INSERT INTO big_decimal_range_test VALUES(-40840124.400000, 1.3);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO big_decimal_range_test VALUES(59016609.300000, 1.3);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
}

void import_subquery_test() {
  const std::string subquery_test("DROP TABLE IF EXISTS subquery_test;");
  run_ddl_statement(subquery_test);
  g_sqlite_comparator.query(subquery_test);
  run_ddl_statement("CREATE TABLE subquery_test(x int) WITH (fragment_size=2);");
  g_sqlite_comparator.query("CREATE TABLE subquery_test(x int);");
  CHECK_EQ(g_num_rows % 2, 0);
  for (ssize_t i = 0; i < g_num_rows; ++i) {
    const std::string insert_query{"INSERT INTO subquery_test VALUES(7);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (ssize_t i = 0; i < g_num_rows / 2; ++i) {
    const std::string insert_query{"INSERT INTO subquery_test VALUES(8);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (ssize_t i = 0; i < g_num_rows / 2; ++i) {
    const std::string insert_query{"INSERT INTO subquery_test VALUES(9);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
}

void import_text_group_by_test() {
  const std::string text_group_by_test("DROP TABLE IF EXISTS text_group_by_test;");
  run_ddl_statement(text_group_by_test);
  run_ddl_statement(
      "CREATE TABLE text_group_by_test(tdef TEXT, tdict TEXT ENCODING DICT, tnone TEXT ENCODING NONE ) WITH "
      "(fragment_size=200);");
  const std::string insert_query{"INSERT INTO text_group_by_test VALUES('hello','world',':-)');"};
  run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
}

void import_join_test() {
  const std::string drop_old_test{"DROP TABLE IF EXISTS join_test;"};
  run_ddl_statement(drop_old_test);
  g_sqlite_comparator.query(drop_old_test);
  const std::string create_test{
      "CREATE TABLE join_test(x int not null, str text encoding dict) WITH (fragment_size=2);"};
  run_ddl_statement(create_test);
  g_sqlite_comparator.query("CREATE TABLE join_test(x int not null, str text);");
  {
    const std::string insert_query{"INSERT INTO join_test VALUES(7, 'foo');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO join_test VALUES(8, 'bar');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
}

void import_emp_table() {
  const std::string drop_old_test{"DROP TABLE IF EXISTS emp;"};
  run_ddl_statement(drop_old_test);
  g_sqlite_comparator.query(drop_old_test);
  const std::string create_test{
      "CREATE TABLE emp(empno INT, ename TEXT ENCODING DICT, deptno INT) WITH (fragment_size=2);"};
  run_ddl_statement(create_test);
  g_sqlite_comparator.query("CREATE TABLE emp(empno INT, ename TEXT ENCODING DICT, deptno INT);");
  {
    const std::string insert_query{"INSERT INTO emp VALUES(1, 'Brock', 10);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO emp VALUES(2, 'Bill', 20);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO emp VALUES(3, 'Julia', 60);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO emp VALUES(4, 'David', 10);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
}

void import_dept_table() {
  const std::string drop_old_test{"DROP TABLE IF EXISTS dept;"};
  run_ddl_statement(drop_old_test);
  g_sqlite_comparator.query(drop_old_test);
  const std::string create_test{"CREATE TABLE dept(deptno INT, dname TEXT ENCODING DICT) WITH (fragment_size=2);"};
  run_ddl_statement(create_test);
  g_sqlite_comparator.query("CREATE TABLE dept(deptno INT, dname TEXT ENCODING DICT);");
  {
    const std::string insert_query{"INSERT INTO dept VALUES(10, 'Sales');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO dept VALUES(20, 'Dev');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO dept VALUES(30, 'Marketing');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO dept VALUES(40, 'HR');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO dept VALUES(50, 'QA');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
}

}  // namespace

TEST(Select, ArrayUnnest) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    unsigned power10 = 1;
    for (const unsigned int_width : {16, 32, 64}) {
      auto result_rows = run_multiple_agg("SELECT COUNT(*), UNNEST(arr_i" + std::to_string(int_width) +
                                              ") AS a FROM array_test GROUP BY a ORDER BY a DESC;",
                                          dt);
      ASSERT_EQ(g_array_test_row_count + 2, result_rows.rowCount());
      ASSERT_EQ(int64_t(g_array_test_row_count + 2) * power10, v<int64_t>(result_rows.getRowAt(0, 1, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows.getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows.getRowAt(0, 0, true)));
      ASSERT_EQ(power10, v<int64_t>(result_rows.getRowAt(g_array_test_row_count + 1, 1, true)));
      power10 *= 10;
    }
    for (const std::string float_type : {"float", "double"}) {
      auto result_rows = run_multiple_agg(
          "SELECT COUNT(*), UNNEST(arr_" + float_type + ") AS a FROM array_test GROUP BY a ORDER BY a DESC;", dt);
      ASSERT_EQ(g_array_test_row_count + 2, result_rows.rowCount());
      ASSERT_EQ(1, v<int64_t>(result_rows.getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows.getRowAt(0, 0, true)));
    }
    {
      auto result_rows =
          run_multiple_agg("SELECT COUNT(*), UNNEST(arr_str) AS a FROM array_test GROUP BY a ORDER BY a DESC;", dt);
      ASSERT_EQ(g_array_test_row_count + 2, result_rows.rowCount());
      ASSERT_EQ(1, v<int64_t>(result_rows.getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows.getRowAt(0, 0, true)));
    }
    {
      auto result_rows =
          run_multiple_agg("SELECT COUNT(*), UNNEST(arr_bool) AS a FROM array_test GROUP BY a ORDER BY a DESC;", dt);
      ASSERT_EQ(size_t(2), result_rows.rowCount());
      ASSERT_EQ(int64_t(g_array_test_row_count * 3), v<int64_t>(result_rows.getRowAt(0, 0, true)));
      ASSERT_EQ(int64_t(g_array_test_row_count * 3), v<int64_t>(result_rows.getRowAt(1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows.getRowAt(0, 1, true)));
      ASSERT_EQ(0, v<int64_t>(result_rows.getRowAt(1, 1, true)));
    }
  }
}

TEST(Select, ArrayIndex) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    for (size_t row_idx = 0; row_idx < g_array_test_row_count; ++row_idx) {
      ASSERT_EQ(
          1,
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE arr_i32[2] = " +
                                        std::to_string(10 * (row_idx + 2)) + " AND x = " + std::to_string(7 + row_idx) +
                                        " AND real_str LIKE 'real_str" + std::to_string(row_idx) + "';",
                                    dt)));
      ASSERT_EQ(
          0,
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE arr_i32[4] > 0 OR arr_i32[4] <= 0;", dt)));
      ASSERT_EQ(
          0,
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE arr_i32[0] > 0 OR arr_i32[0] <= 0;", dt)));
    }
    for (size_t i = 1; i <= 6; ++i) {
      ASSERT_EQ(
          int64_t(g_array_test_row_count / 2),
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE arr_bool[" + std::to_string(i) + "];", dt)));
    }
    ASSERT_EQ(0, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE arr_bool[7];", dt)));
    ASSERT_EQ(0, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE arr_bool[0];", dt)));
    ASSERT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE NOT (arr_i16[7] > 0 AND arr_i16[7] <= 0);", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE NOT (arr_i16[2] > 0 AND arr_i16[2] <= 0);", dt)));
  }
}

TEST(Select, ArrayCountDistinct) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    for (const unsigned int_width : {16, 32, 64}) {
      ASSERT_EQ(int64_t(g_array_test_row_count + 2),
                v<int64_t>(run_simple_agg(
                    "SELECT COUNT(distinct arr_i" + std::to_string(int_width) + ") FROM array_test;", dt)));
      auto result_rows = run_multiple_agg(
          "SELECT COUNT(distinct arr_i" + std::to_string(int_width) + ") FROM array_test GROUP BY x;", dt);
      ASSERT_EQ(g_array_test_row_count, result_rows.rowCount());
      for (size_t row_idx = 0; row_idx < g_array_test_row_count; ++row_idx) {
        ASSERT_EQ(3, v<int64_t>(result_rows.getRowAt(row_idx, 0, true)));
      }
    }
    for (const std::string float_type : {"float", "double"}) {
      ASSERT_EQ(int64_t(g_array_test_row_count + 2),
                v<int64_t>(run_simple_agg("SELECT COUNT(distinct arr_" + float_type + ") FROM array_test;", dt)));
    }
    ASSERT_EQ(int64_t(g_array_test_row_count + 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(distinct arr_str) FROM array_test;", dt)));
    ASSERT_EQ(2, v<int64_t>(run_simple_agg("SELECT COUNT(distinct arr_bool) FROM array_test;", dt)));
  }
}

TEST(Select, ArrayAnyAndAll) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    unsigned power10 = 1;
    for (const unsigned int_width : {16, 32, 64}) {
      ASSERT_EQ(2,
                v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE " + std::to_string(2 * power10) +
                                              " = ANY arr_i" + std::to_string(int_width) + ";",
                                          dt)));
      ASSERT_EQ(int64_t(g_array_test_row_count) - 2,
                v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE " + std::to_string(2 * power10) +
                                              " < ALL arr_i" + std::to_string(int_width) + ";",
                                          dt)));
      power10 *= 10;
    }
    for (const std::string float_type : {"float", "double"}) {
      ASSERT_EQ(
          int64_t(g_array_test_row_count),
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 1 < ANY arr_" + float_type + ";", dt)));
      ASSERT_EQ(
          int64_t(g_array_test_row_count),
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 2 < ANY arr_" + float_type + ";", dt)));
      ASSERT_EQ(
          int64_t(g_array_test_row_count),
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 0 < ALL arr_" + float_type + ";", dt)));
    }
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE x - 5 = ANY arr_i16;", dt)));
    ASSERT_EQ(1, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 'aa' = ANY arr_str;", dt)));
    ASSERT_EQ(2, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 'bb' = ANY arr_str;", dt)));
    ASSERT_EQ(
        int64_t(g_array_test_row_count),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE CAST('t' AS boolean) = ANY arr_bool;", dt)));
    ASSERT_EQ(
        int64_t(0),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE CAST('t' AS boolean) = ALL arr_bool;", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count - 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 'bb' < ALL arr_str;", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count - 1),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 'bb' <= ALL arr_str;", dt)));
    ASSERT_EQ(int64_t(1), v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 'bb' > ANY arr_str;", dt)));
    ASSERT_EQ(int64_t(2), v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 'bb' >= ANY arr_str;", dt)));
    ASSERT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE  real_str = ANY arr_str;", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE  real_str <> ANY arr_str;", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count - 1),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE (NOT ('aa' = ANY arr_str));", dt)));
    // these two test just confirm that the regex does not mess with other similar patterns
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) as SMALL FROM array_test;", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) as COMPANY FROM array_test;", dt)));
  }
}

TEST(Select, ArrayUnsupported) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT MIN(arr_i64) FROM array_test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT UNNEST(arr_str), COUNT(*) cc FROM array_test GROUP BY arr_str;", dt),
                 std::runtime_error);
  }
}

TEST(Select, Joins) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x = test_inner.x;", dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x < test_inner.x + 1;", dt);
    c("SELECT test_inner.x, COUNT(*) AS n FROM test, test_inner WHERE test.x = test_inner.x GROUP BY test_inner.x "
      "ORDER BY n;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.str = test_inner.str;", dt);
    c("SELECT test.str, COUNT(*) FROM test, test_inner WHERE test.str = test_inner.str GROUP BY test.str;", dt);
    c("SELECT test_inner.str, COUNT(*) FROM test, test_inner WHERE test.str = test_inner.str GROUP BY test_inner.str;",
      dt);
    c("SELECT test.str, COUNT(*) AS foobar FROM test, test_inner WHERE test.x = test_inner.x AND test.x > 6 GROUP BY "
      "test.str HAVING foobar > 5;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.real_str LIKE 'real_ba%' AND test.x = test_inner.x;", dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE LENGTH(test.real_str) = 8 AND test.x = test_inner.x;", dt);
    ASSERT_EQ(
        int64_t(g_array_test_row_count / 2 + g_array_test_row_count / 4),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test, test_inner WHERE EXTRACT(HOUR FROM test.m) = 22 AND test.x = test_inner.x;",
            dt)));
    ASSERT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test, test_inner WHERE array_test.arr_i32[array_test.x - 5] = 20 AND "
                  "array_test.x = "
                  "test_inner.x;",
                  dt)));
  }
}

TEST(Select, JoinsAndArrays) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test, array_test_inner WHERE test.x = ALL array_test_inner.arr_i16;", dt)));
    ASSERT_EQ(int64_t(60),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test, array_test_inner WHERE test.x = ANY array_test_inner.arr_i16;", dt)));
    ASSERT_EQ(int64_t(2 * g_array_test_row_count * g_num_rows - 60),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test, array_test_inner WHERE test.x <> ALL array_test_inner.arr_i16;", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test, array_test_inner WHERE 7 = array_test_inner.arr_i16[1];", dt)));
    auto result_rows = run_multiple_agg(
        "SELECT UNNEST(array_test.arr_i16) AS a, test_inner.x, COUNT(*) FROM array_test, test_inner WHERE test_inner.x "
        "= array_test.arr_i16[1] GROUP BY a, test_inner.x;",
        dt);
    ASSERT_EQ(size_t(3), result_rows.rowCount());
  }
}

TEST(Select, OrRewrite) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test WHERE str = 'foo' OR str = 'bar' OR str = 'baz' OR str = 'foo' OR str = 'bar' OR str "
      "= 'baz' OR str = 'foo' OR str = 'bar' OR str = 'baz' OR str = 'baz' OR str = 'foo' OR str = 'bar' OR str = "
      "'baz';",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x = 7 OR x = 8 OR x = 7 OR x = 8 OR x = 7 OR x = 8 OR x = 7 OR x = 8 OR x = 7 "
      "OR x = 8 OR x = 7 OR x = 8;",
      dt);
  }
}

TEST(Select, GpuSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT x, COUNT(*) AS n FROM gpu_sort_test GROUP BY x ORDER BY n DESC;", dt);
    c("SELECT x, COUNT(*), COUNT(*) AS n FROM gpu_sort_test GROUP BY x ORDER BY n DESC;", dt);
  }
}

TEST(Select, GroupByConstrainedByInQueryRewrite) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) AS n, x FROM query_rewrite_test WHERE x IN (2, 5) GROUP BY x HAVING n > 0 ORDER BY n DESC;", dt);
    c("SELECT COUNT(*) AS n, x FROM query_rewrite_test WHERE x IN (2, 99) GROUP BY x HAVING n > 0 ORDER BY n DESC;",
      dt);
    c("SELECT COUNT(*) AS n, str FROM query_rewrite_test WHERE str IN ('str2', 'str5') GROUP BY str HAVING n > 0 ORDER "
      "BY n DESC;",
      dt);
    c("SELECT COUNT(*) AS n, str FROM query_rewrite_test WHERE str IN ('str2', 'str99') GROUP BY str HAVING n > 0 "
      "ORDER BY n DESC;",
      dt);
  }
}

TEST(Select, BigDecimalRange) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT CAST(d AS INT) AS di, COUNT(*) FROM big_decimal_range_test GROUP BY di HAVING di > 0;", dt);
    c("select d1*2 from big_decimal_range_test;", dt);
    c("select 2*d1 from big_decimal_range_test;", dt);
    c("select d1 * (CAST(d1 as INT) + 1) from big_decimal_range_test;", dt);
    c("select (CAST(d1 as INT) + 1) * d1 from big_decimal_range_test;", dt);
  }
}

TEST(Select, AfterDrop) {
  run_ddl_statement("create table droptest (i1 integer);");
  run_multiple_agg("insert into droptest values(1);", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into droptest values(2);", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(3), v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM droptest;", ExecutorDeviceType::CPU)));
  run_ddl_statement("drop table droptest;");
  run_ddl_statement("create table droptest (n1 integer);");
  run_multiple_agg("insert into droptest values(3);", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into droptest values(4);", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(7), v<int64_t>(run_simple_agg("SELECT SUM(n1) FROM droptest;", ExecutorDeviceType::CPU)));
  run_ddl_statement("drop table droptest;");
}

TEST(Select, AfterAlterTableName) {
  run_ddl_statement("create table alter_name_test (i1 integer);");
  run_multiple_agg("insert into alter_name_test values(1);", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into alter_name_test values(2);", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(3), v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM alter_name_test;", ExecutorDeviceType::CPU)));
  run_ddl_statement("alter table alter_name_test rename to alter_name_test_after;");
  run_multiple_agg("insert into alter_name_test_after values(3);", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into alter_name_test_after values(4);", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(10),
            v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM alter_name_test_after;", ExecutorDeviceType::CPU)));
  run_ddl_statement("drop table alter_name_test_after;");
}

TEST(Select, AfterAlterColumnName) {
  run_ddl_statement("create table alter_column_test (i1 integer);");
  run_multiple_agg("insert into alter_column_test values(1);", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into alter_column_test values(2);", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(3), v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM alter_column_test;", ExecutorDeviceType::CPU)));
  run_ddl_statement("alter table alter_column_test rename column i1 to n1;");
  run_multiple_agg("insert into alter_column_test values(3);", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into alter_column_test values(4);", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(10), v<int64_t>(run_simple_agg("SELECT SUM(n1) FROM alter_column_test;", ExecutorDeviceType::CPU)));
  run_ddl_statement("drop table alter_column_test;");
}

TEST(Select, Empty) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM empty;", dt);
    c("SELECT SUM(x) FROM empty;", dt);
    c("SELECT SUM(y) FROM empty;", dt);
    c("SELECT SUM(t) FROM empty;", dt);
    c("SELECT SUM(f) FROM empty;", dt);
    c("SELECT SUM(d) FROM empty;", dt);
    c("SELECT SUM(dd) FROM empty;", dt);
    c("SELECT MIN(x) FROM empty;", dt);
    c("SELECT MIN(y) FROM empty;", dt);
    c("SELECT MIN(t) FROM empty;", dt);
    c("SELECT MIN(f) FROM empty;", dt);
    c("SELECT MIN(d) FROM empty;", dt);
    c("SELECT MIN(dd) FROM empty;", dt);
    c("SELECT MAX(x) FROM empty;", dt);
    c("SELECT MAX(y) FROM empty;", dt);
    c("SELECT MAX(t) FROM empty;", dt);
    c("SELECT MAX(f) FROM empty;", dt);
    c("SELECT MAX(d) FROM empty;", dt);
    c("SELECT MAX(dd) FROM empty;", dt);
    c("SELECT AVG(x) FROM empty;", dt);
    c("SELECT AVG(y) FROM empty;", dt);
    c("SELECT AVG(t) FROM empty;", dt);
    c("SELECT AVG(f) FROM empty;", dt);
    c("SELECT AVG(d) FROM empty;", dt);
    c("SELECT AVG(dd) FROM empty;", dt);
    c("SELECT COUNT(*) FROM test, empty;", dt);
    c("SELECT MIN(ts), MAX(ts) FROM empty;", dt);
    c("SELECT SUM(test.x) FROM test, empty;", dt);
    c("SELECT SUM(test.y) FROM test, empty;", dt);
    c("SELECT SUM(empty.x) FROM test, empty;", dt);
    c("SELECT SUM(empty.y) FROM test, empty;", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 8;", dt);
    c("SELECT SUM(x) FROM test WHERE x > 8;", dt);
    c("SELECT SUM(f) FROM test WHERE x > 8;", dt);
    c("SELECT SUM(d) FROM test WHERE x > 8;", dt);
    c("SELECT SUM(dd) FROM test WHERE x > 8;", dt);
    c("SELECT SUM(dd) FROM empty GROUP BY x, y;", dt);
  }
}

#ifdef HAVE_RAVM
TEST(Select, Subqueries) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT str, SUM(y) AS n FROM test WHERE x > (SELECT COUNT(*) FROM test) - 14 GROUP BY str ORDER BY str ASC;",
      dt);
    c("SELECT COUNT(*) FROM test, (SELECT x FROM test_inner) AS inner_x WHERE test.x = inner_x.x;", dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (SELECT x FROM test WHERE y > 42);", dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (SELECT x FROM test GROUP BY x ORDER BY COUNT(*) DESC LIMIT 1);", dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (SELECT x FROM test GROUP BY x);", dt);
    c("SELECT MIN(yy), MAX(yy) FROM (SELECT AVG(y) as yy FROM test GROUP BY x);", dt);
    c("SELECT COUNT(*) FROM subquery_test WHERE x NOT IN (SELECT x + 1 FROM subquery_test GROUP BY x);", dt);
    c("SELECT MAX(ct) FROM (SELECT COUNT(*) AS ct, str AS foo FROM test GROUP BY foo);", dt);
    c("SELECT COUNT(*) FROM subquery_test WHERE x IN (SELECT x AS foobar FROM subquery_test GROUP BY foobar);", dt);
    c("SELECT * FROM (SELECT x FROM test ORDER BY x);", dt);
    c("SELECT AVG(y) FROM (SELECT * FROM test ORDER BY z LIMIT 5);", dt);
    c("SELECT COUNT(*) FROM subquery_test WHERE x NOT IN (SELECT x + 1 FROM subquery_test GROUP BY x);", dt);
    ASSERT_EQ(int64_t(0), v<int64_t>(run_simple_agg("SELECT * FROM (SELECT rowid FROM test WHERE rowid = 0);", dt)));
    c("SELECT COUNT(*) FROM test WHERE x NOT IN (SELECT x FROM test GROUP BY x ORDER BY COUNT(*));", dt);
    c("SELECT COUNT(*) FROM test WHERE x NOT IN (SELECT x FROM test GROUP BY x);", dt);
    c("SELECT empty.x, CASE WHEN empty.y IN (SELECT empty.y FROM empty GROUP BY empty.y) then empty.y END yy, sum(x) "
      "FROM empty GROUP BY empty.x, yy;",
      dt);
    c("WITH d1 AS (SELECT deptno, dname FROM dept LIMIT 10) SELECT ename, dname FROM emp, d1 WHERE emp.deptno = "
      "d1.deptno LIMIT 10;",
      dt);
  }
}

TEST(Select, InnerJoins) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test JOIN test_inner ON test.x = test_inner.x;", dt);
    c("SELECT y, z FROM test JOIN test_inner ON test.x = test_inner.x order by y;", dt);
#ifdef ENABLE_JOIN_EXEC
    c("SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = c.str;", dt);
    c("SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = c.str JOIN "
      "join_test AS d ON c.x = d.x;",
      dt);
    c("SELECT a.x AS x, y, b.str FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = c.str "
      "ORDER BY x;",
      dt);
    c("SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = c.str WHERE a.y "
      "< 43;",
      dt);
    c("SELECT SUM(a.x), b.str FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = c.str "
      "WHERE a.y "
      "= 43 group by b.str;",
      dt);
    c("SELECT a.str as key0,a.fixed_str as key1,COUNT(*) AS color FROM test a JOIN (select str,count(*) "
      "from test group by str order by COUNT(*) desc limit 40) b on a.str=b.str JOIN (select "
      "fixed_str,count(*) from test group by fixed_str order by count(*) desc limit 40) c on "
      "c.fixed_str=a.fixed_str GROUP BY key0, key1 ORDER BY key0,key1;",
      dt);
#endif
  }
}

TEST(Select, LeftOuterJoins) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT test.x, test_inner.x FROM test LEFT OUTER JOIN test_inner ON test.x = test_inner.x WHERE test.y > 40;",
      dt);
    c("SELECT test.x, test_inner.x FROM test LEFT OUTER JOIN test_inner ON test.x = test_inner.x WHERE test.y > 42;",
      dt);
    c("SELECT test.x, test_inner.x FROM test LEFT OUTER JOIN test_inner ON test.x = test_inner.x;", dt);
    c("SELECT test.str AS foobar, test_inner.str FROM test LEFT OUTER JOIN test_inner ON test.x = test_inner.x WHERE "
      "test.y > 42 ORDER BY foobar DESC LIMIT 8;",
      dt);
    c("SELECT test.x AS foobar, test_inner.x AS inner_foobar, test.f AS f_foobar FROM test LEFT OUTER JOIN test_inner "
      "ON test.str = test_inner.str WHERE test.y > 40 ORDER BY foobar DESC, f_foobar DESC;",
      dt);
    c("SELECT test.str AS foobar, test_inner.str FROM test LEFT OUTER JOIN test_inner ON test.x = test_inner.x WHERE "
      "test_inner.str IS NOT NULL ORDER BY foobar DESC;",
      dt);
    c("SELECT test.x key1, CASE WHEN test_inner.x IS NULL THEN 99 ELSE test_inner.x END key2 FROM test LEFT OUTER JOIN "
      "test_inner ON test.x = test_inner.x GROUP BY key1, key2 ORDER BY key1;",
      dt);
    c("SELECT test_inner.x key1 FROM test LEFT OUTER JOIN test_inner ON test.x = test_inner.x GROUP BY key1 HAVING "
      "key1 IS NOT NULL;",
      dt);
  }
}

TEST(Select, RuntimeFunctions) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT SUM(ABS(-x + 1)) FROM test;", dt);
    c("SELECT SUM(ABS(-y + 1)) FROM test;", dt);
    c("SELECT SUM(ABS(-z + 1)) FROM test;", dt);
    c("SELECT SUM(ABS(-t + 1)) FROM test;", dt);
    c("SELECT SUM(ABS(-dd + 1)) FROM test;", dt);
    c("SELECT SUM(ABS(-f + 1)) FROM test;", dt);
    c("SELECT SUM(ABS(-d + 1)) FROM test;", dt);
    EXPECT_THROW(run_multiple_agg("SELECT MIN(ABS(-ofd + 2)) FROM test;", dt), std::runtime_error);
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(-dd) = -1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(x - 7) = 0;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(x - 7) = 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(x - 8) = -1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(x - 8) = 0;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(y - 42) = 0;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(y - 42) = 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(y - 43) = -1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(y - 43) = 0;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(-f) = -1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(-d) = -1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(ofd) = 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(-ofd) = -1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(ofd) IS NULL;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2 * g_num_rows),
                    v<double>(run_simple_agg("SELECT SUM(SIN(x) * SIN(x) + COS(x) * COS(x)) FROM test;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2 * g_num_rows),
                    v<double>(run_simple_agg("SELECT SUM(SIN(f) * SIN(f) + COS(f) * COS(f)) FROM test;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2 * g_num_rows),
                    v<double>(run_simple_agg("SELECT SUM(SIN(d) * SIN(d) + COS(d) * COS(d)) FROM test;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2 * g_num_rows),
                    v<double>(run_simple_agg("SELECT SUM(SIN(dd) * SIN(dd) + COS(dd) * COS(dd)) FROM test;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2),
                    v<double>(run_simple_agg("SELECT FLOOR(CAST(2.3 AS double)) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<float>(2),
                    v<float>(run_simple_agg("SELECT FLOOR(CAST(2.3 AS float)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg("SELECT FLOOR(CAST(2.3 AS BIGINT)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg("SELECT FLOOR(CAST(2.3 AS SMALLINT)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg("SELECT FLOOR(CAST(2.3 AS INT)) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2), v<double>(run_simple_agg("SELECT FLOOR(2.3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2), v<double>(run_simple_agg("SELECT FLOOR(2.0) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(-3), v<double>(run_simple_agg("SELECT FLOOR(-2.3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(-2), v<double>(run_simple_agg("SELECT FLOOR(-2.0) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(3),
                    v<double>(run_simple_agg("SELECT CEIL(CAST(2.3 AS double)) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<float>(3),
                    v<float>(run_simple_agg("SELECT CEIL(CAST(2.3 AS float)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg("SELECT CEIL(CAST(2.3 AS BIGINT)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg("SELECT CEIL(CAST(2.3 AS SMALLINT)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg("SELECT CEIL(CAST(2.3 AS INT)) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(3), v<double>(run_simple_agg("SELECT CEIL(2.3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2), v<double>(run_simple_agg("SELECT CEIL(2.0) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(-2), v<double>(run_simple_agg("SELECT CEIL(-2.3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(-2), v<double>(run_simple_agg("SELECT CEIL(-2.0) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(
        static_cast<float>(4129511.320307),
        v<double>(run_simple_agg(
            "SELECT DISTANCE_IN_METERS(-74.0059, 40.7217,-122.416667 , 37.783333) FROM test LIMIT 1;", dt)));
  }
}

TEST(Select, UnsupportedPatterns) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
#ifndef ENABLE_JOIN_EXEC
    EXPECT_THROW(run_multiple_agg(
                     "SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON "
                     "b.str = c.str;",
                     dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = "
                     "c.str JOIN join_test AS d ON c.x = d.x;",
                     dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT a.x AS x, y, b.str FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c "
                     "ON b.str = c.str ORDER BY x;",
                     dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = "
                     "c.str WHERE a.y "
                     "< 43;",
                     dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT SUM(a.x), b.str FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON "
                     "b.str = c.str WHERE a.y "
                     "= 43 group by b.str;",
                     dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT a.str as key0,a.fixed_str as key1,COUNT(*) AS color FROM test a JOIN (select str,count(*) "
                     "from test group by str order by COUNT(*) desc limit 40) b on a.str=b.str JOIN (select "
                     "fixed_str,count(*) from test group by fixed_str order by count(*) desc limit 40) c on "
                     "c.fixed_str=a.fixed_str GROUP BY key0, key1 ORDER BY key0,key1;",
                     dt),
                 std::runtime_error);
#endif
  }
}

TEST(Select, TextGroupBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg(
                     " select count(*) from (SELECT tnone, count(*) cc from text_group_by_test group by tnone);", dt),
                 std::runtime_error);
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "select count(*) from (SELECT tdict, count(*) cc from text_group_by_test group by tdict)", dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "select count(*) from (SELECT tdef, count(*) cc from text_group_by_test group by tdef)", dt)));
  }
}

TEST(Select, UnsupportedExtensions) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT TRUNCATE(2016, 1) FROM test LIMIT 1;", dt), std::runtime_error);
  }
}

#endif  // HAVE_RAVM

#ifdef HAVE_CALCITE
TEST(Select, PgShim) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT str, SUM(x), COUNT(str) FROM test WHERE \"y\" = 42 AND str = 'Shim All The Things!' GROUP BY str;", dt);
  }
}

TEST(Select, CaseInsensitive) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT X, COUNT(*) AS N FROM test GROUP BY teSt.x ORDER BY n DESC;", dt);
  }
}
#endif  // HAVE_CALCITE

TEST(Select, DesugarTransform) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT * FROM empty ORDER BY empty.x;", dt);
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << " after initialization";
  testing::InitGoogleTest(&argc, argv);
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("disable-literal-hoisting", "Disable literal hoisting");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);

  if (vm.count("disable-literal-hoisting"))
    g_hoist_literals = false;

  g_session.reset(get_session(BASE_PATH));

  try {
    const std::string drop_old_test{"DROP TABLE IF EXISTS test;"};
    run_ddl_statement(drop_old_test);
    g_sqlite_comparator.query(drop_old_test);
    const std::string create_test{
        "CREATE TABLE test(x int not null, y int, z smallint, t bigint, b boolean, f float, d double, str text "
        "encoding dict, fixed_str text encoding dict(16), real_str text encoding none, m timestamp(0), n time(0), o "
        "date, fx int "
        "encoding fixed(16), dd decimal(10, "
        "2), ss text encoding dict, u int, ofd int, ufd int not null, ofq bigint, ufq bigint not null) WITH "
        "(fragment_size=2);"};
    run_ddl_statement(create_test);
    g_sqlite_comparator.query(
        "CREATE TABLE test(x int not null, y int, z smallint, t bigint, b boolean, f float, d double, str text, "
        "fixed_str text, real_str text, m timestamp(0), n time(0), o date, fx int, dd decimal(10, 2), ss text, u int, "
        "ofd int, ufd int "
        "not null, ofq bigint, ufq bigint not null);");
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'test'";
    return -EEXIST;
  }
  CHECK_EQ(g_num_rows % 2, 0);
  for (ssize_t i = 0; i < g_num_rows; ++i) {
    const std::string insert_query{
        "INSERT INTO test VALUES(7, 42, 101, 1001, 't', 1.1, 2.2, 'foo', 'foo', 'real_foo', '2014-12-13 22:23:15', "
        "'15:13:14', "
        "'1999-09-09', 9, 111.1, 'fish', null, 2147483647, -2147483648, null, -1);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (ssize_t i = 0; i < g_num_rows / 2; ++i) {
    const std::string insert_query{
        "INSERT INTO test VALUES(8, 43, 102, 1002, 'f', 1.2, 2.4, 'bar', 'bar', 'real_bar', '2014-12-13 22:23:15', "
        "'15:13:14', "
        "NULL, NULL, 222.2, null, null, null, -1, 9223372036854775807, -9223372036854775808);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (ssize_t i = 0; i < g_num_rows / 2; ++i) {
    const std::string insert_query{
        "INSERT INTO test VALUES(7, 43, 102, 1002, 't', 1.3, 2.6, 'baz', 'baz', 'real_baz', '2014-12-13 22:23:15', "
        "'15:13:14', "
        "'1999-09-09', 11, 333.3, 'boat', null, 1, -1, 1, -9223372036854775808);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  try {
    const std::string drop_old_array_test{"DROP TABLE IF EXISTS array_test;"};
    run_ddl_statement(drop_old_array_test);
    const std::string create_array_test{
        "CREATE TABLE array_test(x int, arr_i16 smallint[], arr_i32 int[], arr_i64 bigint[], arr_str text[] encoding "
        "dict, "
        "arr_float float[], arr_double double[], arr_bool boolean[], real_str text encoding none) WITH "
        "(fragment_size=2);"};
    run_ddl_statement(create_array_test);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'array_test'";
    return -EEXIST;
  }
  import_array_test("array_test");
  try {
    const std::string drop_old_test{"DROP TABLE IF EXISTS test_inner;"};
    run_ddl_statement(drop_old_test);
    g_sqlite_comparator.query(drop_old_test);
    const std::string create_test{
        "CREATE TABLE test_inner(x int not null, str text encoding dict) WITH (fragment_size=2);"};
    run_ddl_statement(create_test);
    g_sqlite_comparator.query("CREATE TABLE test_inner(x int not null, str text encoding none);");
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'test_inner'";
    return -EEXIST;
  }
  const std::string insert_query{"INSERT INTO test_inner VALUES(7, 'foo');"};
  run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
  g_sqlite_comparator.query(insert_query);
  try {
    const std::string drop_old_array_test{"DROP TABLE IF EXISTS array_test_inner;"};
    run_ddl_statement(drop_old_array_test);
    const std::string create_array_test{
        "CREATE TABLE array_test_inner(x int, arr_i16 smallint[], arr_i32 int[], arr_i64 bigint[], arr_str text[] "
        "encoding "
        "dict, "
        "arr_float float[], arr_double double[], arr_bool boolean[], real_str text encoding none) WITH "
        "(fragment_size=4000000);"};
    run_ddl_statement(create_array_test);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'array_test_inner'";
    return -EEXIST;
  }
  import_array_test("array_test_inner");
  try {
    import_gpu_sort_test();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'gpu_sort_test'";
    return -EEXIST;
  }
  try {
    import_query_rewrite_test();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'query_rewrite_test'";
    return -EEXIST;
  }
  try {
    import_big_decimal_range_test();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'big_decimal_range_test'";
    return -EEXIST;
  }
  try {
    import_subquery_test();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'subquery_test'";
    return -EEXIST;
  }
  try {
    import_text_group_by_test();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'text_group_by_test'";
    return -EEXIST;
  }
  try {
    import_join_test();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'join_test'";
    return -EEXIST;
  }
  try {
    import_emp_table();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'emp'";
    return -EEXIST;
  }
  try {
    import_dept_table();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'dept'";
    return -EEXIST;
  }
  try {
    const std::string drop_old_empty{"DROP TABLE IF EXISTS empty;"};
    run_ddl_statement(drop_old_empty);
    g_sqlite_comparator.query(drop_old_empty);
    const std::string create_empty{
        "CREATE TABLE empty(x int, y int not null, t bigint not null, f float not null, d double not null, dd "
        "decimal(10, 2) not null, ts timestamp);"};
    run_ddl_statement(create_empty);
    g_sqlite_comparator.query(create_empty);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'empty'";
    return -EEXIST;
  }
  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  Executor::nukeCacheOfExecutors();
  const std::string drop_test{"DROP TABLE test;"};
  run_ddl_statement(drop_test);
  g_sqlite_comparator.query(drop_test);
  const std::string drop_test_inner{"DROP TABLE test_inner;"};
  run_ddl_statement(drop_test_inner);
  g_sqlite_comparator.query(drop_test_inner);
  const std::string drop_gpu_sort_test{"DROP TABLE gpu_sort_test;"};
  run_ddl_statement(drop_gpu_sort_test);
  g_sqlite_comparator.query(drop_gpu_sort_test);
  const std::string drop_query_rewrite_test{"DROP TABLE query_rewrite_test;"};
  run_ddl_statement(drop_query_rewrite_test);
  const std::string drop_big_decimal_range_test{"DROP TABLE big_decimal_range_test;"};
  run_ddl_statement(drop_big_decimal_range_test);
  g_sqlite_comparator.query(drop_query_rewrite_test);
  const std::string drop_array_test{"DROP TABLE array_test;"};
  run_ddl_statement(drop_array_test);
  const std::string drop_array_test_inner{"DROP TABLE array_test_inner;"};
  run_ddl_statement(drop_array_test_inner);
  const std::string drop_subquery_test{"DROP TABLE subquery_test;"};
  run_ddl_statement(drop_subquery_test);
  g_sqlite_comparator.query(drop_subquery_test);
  const std::string drop_empty_test{"DROP TABLE empty;"};
  run_ddl_statement(drop_empty_test);
  g_sqlite_comparator.query(drop_empty_test);
  run_ddl_statement("DROP TABLE text_group_by_test;");
  const std::string drop_join_test{"DROP TABLE join_test;"};
  g_sqlite_comparator.query(drop_join_test);
  run_ddl_statement(drop_join_test);
  const std::string drop_emp_table{"DROP TABLE emp;"};
  g_sqlite_comparator.query(drop_emp_table);
  run_ddl_statement(drop_emp_table);
  const std::string drop_dept_table{"DROP TABLE dept;"};
  g_sqlite_comparator.query(drop_dept_table);
  run_ddl_statement(drop_dept_table);
  g_session.reset(nullptr);
  return err;
}

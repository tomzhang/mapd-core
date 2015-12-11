#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <memory>
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include "../Catalog/Catalog.h"
#include "../Parser/parser.h"
#include "../Analyzer/Analyzer.h"
#include "../Parser/ParserNode.h"
#include "../Planner/Planner.h"
#include "../DataMgr/DataMgr.h"
#include "gtest/gtest.h"
#include "glog/logging.h"

using namespace std;
using namespace Catalog_Namespace;
using namespace Analyzer;
using namespace Planner;

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

namespace {
std::unique_ptr<SessionInfo> gsession;

class SQLTestEnv : public ::testing::Environment {
 public:
  virtual void SetUp() {
    boost::filesystem::path base_path{BASE_PATH};
    CHECK(boost::filesystem::exists(base_path));
    auto system_db_file = base_path / "mapd_catalogs" / MAPD_SYSTEM_DB;
    auto data_dir = base_path / "mapd_data";
    UserMetadata user;
    DBMetadata db;
    {
      auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(data_dir.string(), 0,  false, 0);
      if (!boost::filesystem::exists(system_db_file)) {
        SysCatalog syscat(base_path.string(), dataMgr, true);
        syscat.initDB();
      }
      SysCatalog sys_cat(base_path.string(), dataMgr);
      CHECK(sys_cat.getMetadataForUser(MAPD_ROOT_USER, user));
      if (!sys_cat.getMetadataForUser("gtest", user)) {
        sys_cat.createUser("gtest", "test!test!", false);
        CHECK(sys_cat.getMetadataForUser("gtest", user));
      }
      if (!sys_cat.getMetadataForDB("gtest_db", db)) {
        sys_cat.createDatabase("gtest_db", user.userId);
        CHECK(sys_cat.getMetadataForDB("gtest_db", db));
      }
    }
    auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(data_dir.string(), 0, false, 0);
    gsession.reset(new SessionInfo(std::make_shared<Catalog_Namespace::Catalog>(base_path.string(), db, dataMgr),
                                   user,
                                   ExecutorDeviceType::GPU,
                                   0));
  }
};

void run_ddl(const string& input_str) {
  SQLParser parser;
  list<Parser::Stmt*> parse_trees;
  string last_parsed;
  CHECK_EQ(parser.parse(input_str, parse_trees, last_parsed), 0);
  CHECK_EQ(parse_trees.size(), size_t(1));
  auto stmt = parse_trees.front();
  unique_ptr<Stmt> stmt_ptr(stmt);  // make sure it's deleted
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
  CHECK(ddl != nullptr);
  ddl->execute(*gsession);
}

RootPlan* plan_dml(const string& input_str) {
  SQLParser parser;
  list<Parser::Stmt*> parse_trees;
  string last_parsed;
  CHECK_EQ(parser.parse(input_str, parse_trees, last_parsed), 0);
  CHECK_EQ(parse_trees.size(), size_t(1));
  auto stmt = parse_trees.front();
  unique_ptr<Stmt> stmt_ptr(stmt);  // make sure it's deleted
  Parser::DMLStmt* dml = dynamic_cast<Parser::DMLStmt*>(stmt);
  CHECK(dml != nullptr);
  Query query;
  dml->analyze(gsession->get_catalog(), query);
  Optimizer optimizer(query, gsession->get_catalog());
  RootPlan* plan = optimizer.optimize();
  return plan;
}
}  // namespace

TEST(ParseAnalyzePlan, Create) {
  ASSERT_NO_THROW(run_ddl(
                      "create table if not exists fat (a boolean, b char(5), c varchar(10), d numeric(10,2) "
                      "encoding rl, e decimal(5,3) encoding sparse(16), f int encoding fixed(16), g smallint, "
                      "h real, i float, j double, k bigint encoding diff, l text not null encoding dict, m "
                      "timestamp(0), n time(0), o date);"););
  ASSERT_TRUE(gsession->get_catalog().getMetadataForTable("fat") != nullptr);
  ASSERT_NO_THROW(run_ddl("create table if not exists skinny (a smallint, b int, c bigint);"););
  ASSERT_TRUE(gsession->get_catalog().getMetadataForTable("skinny") != nullptr);
  ASSERT_NO_THROW(run_ddl(
                      "create table if not exists smallfrag (a int, b text, c bigint) with "
                      "(fragment_size = 1000, page_size = 512);"););
  const TableDescriptor* td = gsession->get_catalog().getMetadataForTable("smallfrag");
  EXPECT_TRUE(td->maxFragRows == 1000 && td->fragPageSize == 512);
  ASSERT_NO_THROW(run_ddl(
                      "create table if not exists testdict (a varchar(100) encoding dict(8), c "
                      "text encoding dict);"););
  td = gsession->get_catalog().getMetadataForTable("testdict");
  const ColumnDescriptor* cd = gsession->get_catalog().getMetadataForColumn(td->tableId, "a");
  const DictDescriptor* dd = gsession->get_catalog().getMetadataForDict(cd->columnType.get_comp_param());
  ASSERT_TRUE(dd != nullptr);
  EXPECT_EQ(dd->dictNBits, 8);
}

TEST(ParseAnalyzePlan, Select) {
  EXPECT_NO_THROW({ unique_ptr<RootPlan> plan_ptr(plan_dml("select * from fat;")); });
  EXPECT_NO_THROW({ unique_ptr<RootPlan> plan_ptr(plan_dml("select f.* from fat f;")); });
  EXPECT_NO_THROW({ unique_ptr<RootPlan> plan_ptr(plan_dml("select cast(a as int), d, l from fat;")); });
  EXPECT_NO_THROW({ unique_ptr<RootPlan> plan_ptr(plan_dml("select -1, -1.1, -1e-3, -a from fat;")); });
  EXPECT_NO_THROW({ unique_ptr<RootPlan> plan_ptr(plan_dml("select a, d, l from fat where not 1=0;")); });
  EXPECT_NO_THROW({ unique_ptr<RootPlan> plan_ptr(plan_dml("select b, d+e, f*g as y from fat;")); });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select b, d+e, f*g as y from fat order by 2 asc null last, 3 desc null first;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select b, d+e, f*g as y from fat order by 2 asc null last, 3 desc null first limit 10;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select b, d+e, f*g as y from fat order by 2 asc null last, 3 desc null first limit all "
        "offset 100 rows;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select a, d, g from fat where f > 100 and g is null and k <= 100000000000 and c = "
        "'xyz';"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select a, d, g from fat where f > 100 and g is not null or k <= 100000000000 and c = "
        "'xyz';"));
  });
  EXPECT_NO_THROW({ unique_ptr<RootPlan> plan_ptr(plan_dml("select i, j, k from fat where l like '%whatever%';")); });
  EXPECT_NO_THROW(
      { unique_ptr<RootPlan> plan_ptr(plan_dml("select i, j, k from fat where l like '%whatever@%_' escape '@';")); });
  EXPECT_NO_THROW(
      { unique_ptr<RootPlan> plan_ptr(plan_dml("select i, j, k from fat where l ilike '%whatever@%_' escape '@';")); });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml("select i, j, k from fat where l not like '%whatever@%_' escape '@';"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml("select i, j, k from fat where l not ilike '%whatever@%_' escape '@';"));
  });
  EXPECT_NO_THROW(
      { unique_ptr<RootPlan> plan_ptr(plan_dml("select e, f, g from fat where e in (3.5, 133.33, 222.22);")); });
  EXPECT_NO_THROW(
      { unique_ptr<RootPlan> plan_ptr(plan_dml("select e, f, g from fat where e not in (3.5, 133.33, 222.22);")); });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select e, f, g from fat where e not in (3.5, 133.33, 222.22) or l not like "
        "'%whatever%';"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select a, b, c from fat where i between 10e5 and 10e6 and j not between 10e-4 and "
        "10e-1;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select case when e between 10 and 20 then 1 when e between 20 and 40 then 2 when e is "
        "null then 100 else 5 end as x, a from fat where case when g > f then 100 when l like "
        "'%whatever%' then 200 else 300 end > 100;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select case when e between 10 and 20 then 1 when e between 20 and 40 then 2.1 when e is "
        "null then 100.33 else 5e2 end as x, a from fat where case when g > f then 100 when l like "
        "'%whatever%' then 200 else 300 end > 100;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select case when e between 10 and 20 then i when e between 20 and 40 then j when e is "
        "null then d else 5e2 end as x, a from fat where case when g > f then 100 when l like "
        "'%whatever%' then 200 else 300 end > 100;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select count(*), min(a), max(a), avg(b), sum(c), count(distinct b) from skinny;"));
  });
  EXPECT_NO_THROW({ unique_ptr<RootPlan> plan_ptr(plan_dml("select a+b as x from skinny group by x;")); });
  EXPECT_NO_THROW({ unique_ptr<RootPlan> plan_ptr(plan_dml("select a, b, count(*) from skinny group by a, b;")); });
  EXPECT_NO_THROW(
      { unique_ptr<RootPlan> plan_ptr(plan_dml("select c, avg(b) from skinny where a > 10 group by c;")); });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(
        plan_dml("select c, avg(b) from skinny where a > 10 group by c having max(a) < 100;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select c, avg(b) from skinny where a > 10 group by c having max(a) < 100 and count(*) > "
        "1000;"));
  });
  EXPECT_NO_THROW({ unique_ptr<RootPlan> plan_ptr(plan_dml("select count(*)*avg(c) - sum(c) from skinny;")); });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select a+b as x, count(*)*avg(c) - sum(c) as y from skinny where c between 100 and 200 "
        "group by a, b;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select a+b as x, count(*)*avg(c) - sum(c) as y from skinny where c between 100 and 200 "
        "group by a, b having b > 2*a and min(b) > max(a);"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select a+b as x, count(*)*avg(c) - sum(c) as y from skinny where c between 100 and 200 "
        "group by a, b order by x desc null first;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select a+b as x, count(*)*avg(c) - sum(c) as y from skinny where c between 100 and 200 "
        "group by a, b order by x desc null first limit 10 offset 100000000;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select cast(a+b as decimal(10,3)) as x, count(*)*avg(c) - sum(c) as y from skinny where c "
        "between 100 and 200 group by a, b order by x desc null first limit 10 offset 100000000;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select a+b as x, count(*)*avg(c) - sum(c) as y from skinny where c between 100 and 200 "
        "group by x, b having x > 10;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select distinct a+b as x, count(*)*avg(c) - sum(c) as y from skinny where c between 100 "
        "and 200 group by x, b having x > 10;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select * from fat where m < timestamp(0) '2015-02-18 13:15:55' and n >= time(0) '120000' "
        "and o <> date '05/06/2014';"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select extract(year from date '2015-02-28'), extract(month from date '2014-12-13'), "
        "extract(day from timestamp(0) '1998-10-24 03:14:55'), extract(dow from date "
        "'1936-02-09'), extract(doy from timestamp(0) '2015-02-18 01:02:11'), extract(hour from "
        "time(0) '111233'), extract(minute from m), extract(second from n), extract(epoch from o) "
        "from fat where cast(timestamp(0) '2015-02-18 12:13:14' as int) > 1000;"));
  });
  EXPECT_NO_THROW({
    unique_ptr<RootPlan> plan_ptr(plan_dml(
        "select * from fat where m >= '1999-09-09T111111' and n <= '222222' and o = "
        "'1996-02-23';"));
  });
}

TEST(ParseAnalyzePlan, Insert) {
  EXPECT_NO_THROW(
      { unique_ptr<RootPlan> plan_ptr(plan_dml("insert into skinny values (12345, 100000000, 100000000000);")); });
  EXPECT_NO_THROW({ unique_ptr<RootPlan> plan_ptr(plan_dml("insert into skinny select 2*a, 2*b, 2*c from skinny;")); });
}

TEST(ParseAnalyzePlan, Views) {
  EXPECT_NO_THROW(run_ddl("create view if not exists voo as select * from skinny where a > 15;"););
  EXPECT_NO_THROW(run_ddl("create materialized view if not exists moo as select * from skinny where a > 15;"););
  EXPECT_NO_THROW(run_ddl(
                      "create materialized view if not exists goo with (storage = 'gpu', refresh = "
                      "'auto') as select * from skinny where a > 15;"););
  EXPECT_NO_THROW(run_ddl(
                      "create materialized view if not exists mic (col, avg_b) with (storage = 'mic', refresh "
                      "= 'manual') as select c, avg(b) from skinny where a > 10 group by c;"););
  EXPECT_NO_THROW(run_ddl(
                      "create materialized view if not exists fatview with (storage = 'cpu', "
                      "refresh = 'immediate') as select a, d, g from fat where f > 100 and g is "
                      "not null or k <= 100000000000 and c = 'xyz';"););
  EXPECT_NO_THROW(run_ddl("refresh materialized view fatview;"););
  EXPECT_NO_THROW({ unique_ptr<RootPlan> plan_ptr(plan_dml("select * from fatview;")); });
}

TEST(ParseAnalyzePlan, Drop) {
  EXPECT_NO_THROW(run_ddl("drop view if exists voo;"));
  EXPECT_NO_THROW(run_ddl("drop view if exists moo;"));
  EXPECT_NO_THROW(run_ddl("drop view if exists goo;"));
  EXPECT_NO_THROW(run_ddl("drop view if exists mic;"));
  EXPECT_NO_THROW(run_ddl("drop view if exists fatview;"));
  EXPECT_NO_THROW(run_ddl("drop table if exists fat;"));
  EXPECT_NO_THROW(run_ddl("drop table if exists skinny;"));
  EXPECT_NO_THROW(run_ddl("drop table if exists smallfrag;"));
  EXPECT_NO_THROW(run_ddl("drop table if exists testdict;"));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SQLTestEnv);
  return RUN_ALL_TESTS();
}

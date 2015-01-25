/**
 * @file	TablePartitionerMgrTest 
 * @author	Todd Mostak <todd@mapd.com>
 */

#include "gtest/gtest.h"
#include "../../DataMgr/DataMgr.h"
#include "../TablePartitionerMgr.h"
#include "../InsertOrderTablePartitioner.h"

#include <iostream>
#include <vector>
#include <math.h>
#include <boost/filesystem.hpp>

using namespace std;

#define TABLEPARTITIONERMGR_UNIT_TESTING

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

namespace Partitioner_Namespace {

    class TablePartitionerMgrTest : public ::testing::Test {
        protected:
            virtual void SetUp() {
                maxPartitionRows = 1000000;
                deleteData("data");
                dataMgr = new Data_Namespace::DataMgr(2,"data");
                ChunkKey chunkKeyPrefix = {0,1,2};

                tableDesc0.tableId = 0;
                tableDesc0.tableName = "table_0";
                tableDesc0.nColumns = 2;

                tableDesc1.tableId = 1;
                tableDesc1.tableName = "table_1";
                tableDesc1.nColumns = 3;

                columnDescsTable0.push_back(new ColumnDescriptor(0,0,"col0_0",kINT,kENCODING_FIXED,8)); 
                columnDescsTable0.push_back(new ColumnDescriptor(0,1,"col0_1",kFLOAT,kENCODING_NONE)); 

                columnDescsTable1.push_back(new ColumnDescriptor(1,0,"col1_0",kDOUBLE,kENCODING_NONE)); 
                columnDescsTable1.push_back(new ColumnDescriptor(1,1,"col1_1",kBIGINT,kENCODING_FIXED,32)); 
                columnDescsTable1.push_back(new ColumnDescriptor(1,2,"col1_2",kSMALLINT,kENCODING_NONE)); 

                tablePartitionerMgr = new TablePartitionerMgr(dataMgr);
            }

            virtual void TearDown() {
                delete tablePartitionerMgr;
                delete dataMgr;
                for (auto colDesc0It = columnDescsTable0.begin(); colDesc0It != columnDescsTable0.end(); ++ colDesc0It) { 
                    delete *colDesc0It;
                }
                for (auto colDesc1It = columnDescsTable1.begin(); colDesc1It != columnDescsTable1.end(); ++ colDesc1It) { 
                    delete *colDesc1It;
                }
            }

            void deleteData(const std::string &dirName) {
                boost::filesystem::remove_all(dirName);
            }

            Data_Namespace::DataMgr *dataMgr;
            TablePartitionerMgr *tablePartitionerMgr;
            InsertOrderTablePartitioner *insertOrderTablePartitioner;
            TableDescriptor  tableDesc0;
            TableDescriptor  tableDesc1;
            vector <ColumnDescriptor *> columnDescsTable0;
            vector <ColumnDescriptor *> columnDescsTable1;
            int64_t maxPartitionRows;
    };

    TEST_F (TablePartitionerMgrTest, createPartitionerTest) {
        tablePartitionerMgr->createPartitionerForTable(0,tableDesc0,colDescsTable0);
    }

} // Partitioner_Namespace

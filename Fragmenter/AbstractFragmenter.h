/**
 * @file    AbstractFragmenter.h
 * @author  Todd Mostak <todd@map-d.com
 */

#ifndef _ABSTRACT_FRAGMENTER_H
#define _ABSTRACT_FRAGMENTER_H

#include "../Shared/sqltypes.h"
#include "Fragmenter.h"
#include <vector>
#include <string>

// Should the ColumnInfo and FragmentInfo structs be in
// AbstractFragmenter?

namespace Data_Namespace {
    class AbstractBuffer;
    class AbstractDataMgr;
};

namespace Fragmenter_Namespace {



/*
 * @type AbstractFragmenter
 * @brief abstract base class for all table partitioners
 *
 * The virtual methods of this class provide an interface
 * for an interface for getting the id and type of a 
 * partitioner, inserting data into a partitioner, and
 * getting the partitions (fragments) managed by a
 * partitioner that must be queried given a predicate
 */

class AbstractFragmenter { 

    public:
        virtual ~AbstractFragmenter() {}

        /**
         * @brief Should get the partitions(fragments) 
         * where at least one tuple could satisfy the
         * (optional) provided predicate, given any 
         * statistics on data distribution the partitioner
         * keeps. May also prune the predicate.
         */

        //virtual void getFragmentsForQuery(QueryInfo &queryInfo, const void *predicate = 0) = 0;
        virtual void getFragmentsForQuery(QueryInfo &queryInfo) = 0;

        /**
         * @brief Given data wrapped in an InsertData struct,
         * inserts it into the correct partitions
         */

        virtual void insertData (const InsertData &insertDataStruct) = 0;

        /**
         * @brief Gets the id of the partitioner
         */
        virtual int getFragmenterId() = 0;

        /**
         * @brief Gets the string type of the partitioner
         * @todo have a method returning the enum type?
         */

        virtual std::string getFragmenterType() = 0;

};

} // Fragmenter_Namespace

#endif // _ABSTRACT_FRAGMENTER_H

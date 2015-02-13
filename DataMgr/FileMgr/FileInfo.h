#ifndef FILEINFO_H
#define FILEINFO_H

#include "../../Shared/global.h"
#include "../../Shared/types.h"
#include "Page.h"
#include <cstdio> 
#include <set> 
#include <vector> 
#include <mutex>
#include <unistd.h> 
#include <fcntl.h>

namespace File_Namespace {
    
    struct Page;

    /**
     * @type FileInfo
     * @brief A FileInfo type has a file pointer and metadata about a file.
     *
     * A file info structure wraps around a file pointer in order to contain additional
     * information/metadata about the file that is pertinent to the file manager.
     *
     * The free pages (freePages) within a file must be tracked, and this is implemented using a
     * basic STL set. The set ensures that no duplicate pages are included, and that the pages
     * are sorted, faciliating the obtaining of consecutive free pages by a constant time
     * pop operation, which may reduce the cost of DBMS disk accesses.
     *
     * Helper functions are provided: size(), available(), and used().
     */
    struct FileInfo {
        int fileId;							/// unique file identifier (i.e., used for a file name)
        FILE *f;							/// file stream object for the represented file
        size_t pageSize;				/// the fixed size of each page in the file
        size_t numPages;				/// the number of pages in the file
        //std::vector<Page*> pages;			/// Page pointers for each page (including free pages)
        std::set<size_t> freePages; 	/// set of page numbers of free pages
        std::mutex freePagesMutex_;  
        
        /// Constructor
        FileInfo(const int fileId, FILE *f, const size_t pageSize, const size_t numPages, const bool init = false);

        
        /// Destructor
        ~FileInfo();

        /// Adds all pages to freePages and zeroes first four bytes of header
        //for each apge
        void initNewFile();

        void freePage(int pageId);
        int getFreePage();
            
        void openExistingFile(std::vector<HeaderInfo> &headerVec, const int fileMgrEpoch);
        /// Prints a summary of the file to stdout
        void print(bool pagesummary);
        
        /// Returns the number of bytes used by the file
        inline size_t size() {
            return pageSize * numPages;
        }

        inline int syncToDisk() {
            return fflush(f);
            //return fcntl(fileno(f),51);

        }
        
        /// Returns the number of free bytes available
        inline size_t available() {
            return freePages.size() * pageSize;
        }
        
        /// Returns the number of free pages available
        inline size_t numFreePages() {
            std::lock_guard < std::mutex > lock (freePagesMutex_);
            return freePages.size();
        }
        
        /// Returns the amount of used bytes; size() - available()
        inline size_t used() {
            return size() - available();
        }
    };
} // File_Namespace

#endif // kkkkk 

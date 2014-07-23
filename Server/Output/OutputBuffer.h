#ifndef OUTPUT_BUFFER_H
#define OUTPUT_BUFFER_H

#include <vector>
#include <list>
#include <queue>
#include <string>

class OutputBuffer {

    private:
        std::queue<std::vector <char>, std::list <std::vector <char> > > dataQueue_;
        bool isFinal_;

    public:

        OutputBuffer (): isFinal_(false) {}

        inline void addSubBuffer () {
            writeLastSubBufferSize();
            dataQueue_.push(std::vector <char> (4)); // Start with 4 so we have space later to write in size
        }
        
        inline void finalize() {
            writeLastSubBufferSize();
            isFinal_ = true;
            dataQueue_.push(std::vector <char> (4)); // Start with 4 so we have space later to write in size
            writeLastSubBufferSize();
            // if this was multithreaded would notify consumer this was last
            // element
        }

        void writeLastSubBufferSize();

        void appendData (const void *data, const size_t size);

        void appendData (const char *data, const size_t size);// copies c-style string (not null-terminated)

        void appendData (const std::string &data);

        template <typename T> void appendData (T data) { // must leave in header file because templated
            size_t dataSize = sizeof(T);
            char * dataCharPtr = reinterpret_cast <char *> (&data);
            dataQueue_.back().insert(dataQueue_.back().end(), dataCharPtr, dataCharPtr + dataSize);
        }

        template <typename T> void writeDataAtPos (T data, const size_t offset) { // must leave in header file because templated
            size_t dataSize = sizeof(T);
            char * dataCharPtr = reinterpret_cast <char *> (&data);
            std::copy(dataCharPtr,dataCharPtr+4,dataQueue_.back().begin() + offset);
        }


        inline bool empty () const {
            return dataQueue_.size() == 0;
        }

        inline size_t size() const {
            return dataQueue_.size();
        }

        inline const std::vector <char>& front() const {
            return dataQueue_.front();
        }

        inline void pop() {
            dataQueue_.pop();
        }
};

#endif // OUTPUT_BUFFER_H

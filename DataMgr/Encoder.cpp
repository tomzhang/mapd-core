#include "Encoder.h"
#include "NoneEncoder.h"
#include "FixedLengthEncoder.h"
#include "StringNoneEncoder.h"
#include "StringTokDictEncoder.h"
#include <glog/logging.h>


Encoder * Encoder::Create(Data_Namespace::AbstractBuffer *buffer, const SQLTypeInfo sqlType) {
    switch (sqlType.get_compression()) {
        case kENCODING_NONE: {
            switch (sqlType.get_type()) {
                case kBOOLEAN: {
                    return new NoneEncoder <int8_t>  (buffer);
                    break;
                }
                case kSMALLINT: {
                    return new NoneEncoder <int16_t>  (buffer);
                    break;
                }
                case kINT: {
                    return new NoneEncoder <int32_t>  (buffer);
                    break;
                }
                case kBIGINT: 
								case kNUMERIC: 
								case kDECIMAL: {
                    return new NoneEncoder <int64_t>  (buffer);
                    break;
                }
                case kFLOAT: {
                    return new NoneEncoder <float>  (buffer);
                    break;
                }
                case kDOUBLE: {
                    return new NoneEncoder <double>  (buffer);
                    break;
                }
								case kTEXT:
								case kVARCHAR:
								case kCHAR:
									return new StringNoneEncoder(buffer);
								case kTIME:
								case kTIMESTAMP:
								case kDATE:
									return new NoneEncoder<time_t>(buffer);
                default: {
                    return 0;
                }
            }
            break;
         }
        case kENCODING_FIXED: {
            switch (sqlType.get_type()) {
                case kSMALLINT: {
                    switch(sqlType.get_comp_param()) {
                        case 8:
                            return new FixedLengthEncoder <int16_t,int8_t>  (buffer);
                            break;
                        case 16:
                            return new NoneEncoder <int16_t> (buffer);
                            break;
                        default:
                            return 0;
                            break;
                    }
                break;
                }
                case kINT: {
                    switch(sqlType.get_comp_param()) {
                        case 8:
                            return new FixedLengthEncoder <int32_t,int8_t> (buffer);
                            break;
                        case 16:
                            return new FixedLengthEncoder <int32_t,int16_t> (buffer);
                            break;
                        case 32:
                            return new NoneEncoder <int32_t> (buffer);
                            break;
                        default:
                            return 0;
                            break;
                    }
                }
                break;
                case kBIGINT: 
								case kNUMERIC:
								case kDECIMAL: {
                    switch(sqlType.get_comp_param()) {
                        case 8:
                            return new FixedLengthEncoder <int64_t,int8_t> (buffer);
                            break;
                        case 16:
                            return new FixedLengthEncoder <int64_t,int16_t> (buffer);
                            break;
                        case 32:
                            return new FixedLengthEncoder <int64_t,int32_t> (buffer);
                            break;
                        case 64:
                            return new NoneEncoder <int64_t> (buffer);
                            break;
                        default:
                            return 0;
                            break;
                    }
                break;
                }
								case kTIME:
								case kTIMESTAMP:
								case kDATE:
									assert(false);
									break;
                default: {
                    return 0;
                    break;
                }
            } // switch (sqlType)
            break;
        } // Case: kENCODING_FIXED
        case kENCODING_DICT: {
          CHECK(sqlType.is_string());
          return new NoneEncoder <int32_t> (buffer);
          break;
        }
        case kENCODING_TOKDICT:
          CHECK(sqlType.is_string());
          switch (sqlType.get_elem_size()) {
            case 1:
              return new StringTokDictEncoder<int8_t>(buffer);
            case 2:
              return new StringTokDictEncoder<int16_t>(buffer);
            case 4:
              return new StringTokDictEncoder<int32_t>(buffer);
            default:
              assert(false);
          }
          break;
        default: {
            return 0;
            break;
        }
    } // switch (encodingType)
    return 0;

}

void Encoder::getMetadata(ChunkMetadata &chunkMetadata) {
    //chunkMetadata = metadataTemplate_; // invoke copy constructor
    chunkMetadata.sqlType = buffer_ -> sqlType; 
    chunkMetadata.numBytes = buffer_ -> size();
    chunkMetadata.numElements = numElems;
}

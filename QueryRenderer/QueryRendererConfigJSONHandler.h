#ifndef RENDER_CONFIG_JSON_HANDLER_H_
#define RENDER_CONFIG_JSON_HANDLER_H_

#include "rapidjson/reader.h"
#include "rapidjson/document.h"
#include "rapidjson/allocators.h"
#include <iostream>
#include <string>
#include <map>
#include <limits>  // std::numeric_limits
#include "QueryRendererConfig.h"
#include "ScaleConfig.h"

using namespace rapidjson;

namespace MapD_Renderer {

struct ScaleObject {
    std::string name;
    std::string type;
};

struct QueryRendererConfigJSONHandler : public BaseReaderHandler<UTF8<>, QueryRendererConfigJSONHandler> {
    QueryRendererConfigJSONHandler() : _stateStack({kExpectObjectStart}), _rendererConfig(), _isEclecticArray(false), _array(kArrayType), _intRef(nullptr), _uintRef(nullptr), _strRef(nullptr) {}

    bool StartObject() {
        switch (_stateStack.back()) {
            case kExpectObjectStart:
            case kExpectArrayItemOrArrayEnd:
                _stateStack.push_back(kExpectKeyOrObjectEnd);
                return true;
            default:
                return false;
        }


        // switch (_state) {
        //     case kExpectObjectStart:
        //         _state = kExpectKeyOrObjectEnd;
        //         return true;
        //     case kExpectScaleObjectStart:
        //         _state = kExpectScaleKeyOrObjectEnd;
        //         return true;
        //     default:
        //         return false;
        // }
        return false;
    }

    bool Key(const Ch* str, SizeType length, bool copy) {
        _key = std::string(str, length);

        switch (_stateStack.back()) {
            case kExpectKeyOrObjectEnd:
                if (_key == "width") {
                    _stateStack.push_back(kExpectIntValue);
                    _intRef = &_rendererConfig.width;
                    return true;
                } else if (_key == "height") {
                    _stateStack.push_back(kExpectIntValue);
                    _intRef = &_rendererConfig.height;
                    return true;
                } else if (_key == "scales") {
                    _stateStack.push_back(kExpectArrayStart);
                    return true;
                } else if (_key == "name") {
                    _stateStack.push_back(kExpectStringValue);
                    _strRef = &_scaleObj.name;
                    return true;
                } else if (_key == "type") {
                    _stateStack.push_back(kExpectStringValue);
                    _strRef = &_scaleObj.type;
                    _expectedStrVals.assign({"linear", "ordinal"});
                    return true;
                } else if (_key == "domain") {
                    _stateStack.push_back(kExpectArrayStart);
                    _isEclecticArray = false;
                    return true;
                } else {
                    return false;
                }
            default:
                return false;
        }

        // switch (_state) {
        //     case kExpectKeyOrObjectEnd:
        //         if (_key == "width") {
        //             _state = kExpectObjectIntValue;
        //             _intRef = &_rendererConfig.width;
        //             return true;
        //         } else if (_key == "height") {
        //             _state = kExpectObjectIntValue;
        //             _intRef = &_rendererConfig.height;
        //             return true;
        //         } else if (_key == "scales") {
        //             _state = kExpectScaleArray;
        //             return true;
        //         } else {
        //             return false;
        //         }
        //     case kExpectScaleKeyOrObjectEnd:
        //         if (_key == "name") {
        //             _state = kExpectScaleName;
        //             return true;
        //         } else if (_key == "type") {
        //             _state = kExpectScaleType;
        //             return true;
        //         } else if (_key == "domain") {
        //             // _state = kExpectScaleDomain;
        //             _state = kExpectArrayStart;
        //             return true;
        //         } else if (_key == "range") {
        //             _state = kExpectArrayStart;
        //             return true;
        //         } else {
        //             return false;
        //         }
        //     default:
        //         return false;
        // }
    }

    bool String(const char* str, SizeType length, bool copy) {
        bool rtn = false;

        _strVal = std::string(str, length);

        if (_strRef != nullptr) {
            switch (_stateStack.back()) {
                case kExpectStringValue:
                    rtn = (_expectedStrVals.size() > 0 ? false : true);

                    for (auto &expectedVal : _expectedStrVals) {
                        if (_strVal == expectedVal) {
                            rtn = true;
                            break;
                        }
                    }

                    if (rtn) {
                        (*_strRef) = _strVal;
                    }

                    break;
                default:
                    rtn = false;
            }
        }

        if (rtn) {
            _stateStack.pop_back();
        }

        _expectedStrVals.clear();

        return rtn;

        // switch (_state) {
        //     case kExpectScaleName:
        //         _scaleObj.name = _strVal;
        //         _state = kExpectScaleKeyOrObjectEnd;
        //         return true;
        //     case kExpectScaleType:
        //         _scaleObj.type = _strVal;
        //         _state = kExpectScaleKeyOrObjectEnd;
        //         if (_strVal == "linear") {
        //             return true;
        //         } else if (_strVal == "ordinal") {
        //             return true;
        //         } else {
        //             return false;
        //         }
        //     default:
        //         return false;
        // }

    }

    bool Int(int i) {
        bool rtn = false;

        if (_intRef != nullptr) {
            switch (_stateStack.back()) {
                case kExpectIntValue:
                    (*_intRef) = i;
                    rtn = true;
                    break;
                default:
                    rtn = false;
                    break;
            }

            _intRef = nullptr;
        }

        if (rtn) {
            _stateStack.pop_back();
        }

        return rtn;

        // switch (_state) {
        //     case kExpectObjectIntValue:
        //         (*_intRef) = i;
        //         _state = kExpectKeyOrObjectEnd;
        //         return true;
        //     default:
        //         return false;
        // }
    }

    bool Uint(unsigned int i) {
        bool rtn = false;

        switch (_stateStack.back()) {
            case kExpectUIntValue:
                if (_uintRef != nullptr) {
                    (*_uintRef) = i;
                    rtn = true;
                }
                break;
            case kExpectIntValue:
                if (_intRef != nullptr &&
                    i <= std::numeric_limits<int>::max()) {
                    (*_intRef) = static_cast<int>(i);
                    rtn = true;
                }
                break;
            default:
                rtn = false;
                break;
        }

        if (rtn) {
            _stateStack.pop_back();
        }

        return rtn;

        // switch (_state) {
        //     case kExpectObjectIntValue:
        //         (*_intRef) = static_cast<int>(i);
        //         _state = kExpectKeyOrObjectEnd;
        //         return true;
        //     default:
        //         return false;
        // }
    }

    bool Double(double d) {
        // switch(_state) {
        //     case kExpectScaleDomainItem:
        //         _arrayVal.SetDouble(d);
        //         if (_array.Size() && !_array[0].IsDouble()) {
        //             return false;
        //         }
        //         _array.PushBack(_arrayVal, allocator);
        //         return true;
        //     default:
        //         return false;
        // }
        return false;
    }

    bool StartArray() {
        // switch (_state) {
        //     default:
        //         return false;
        // }

        switch (_stateStack.back()) {
            case kExpectArrayStart:
                _stateStack.push_back(kExpectArrayItemOrArrayEnd);
                return true;
            default:
                return false;
        }
    }

    bool EndArray(SizeType elementCount) {
        _array.Clear();

        if (_stateStack.back() != kExpectArrayItemOrArrayEnd) {
            return false;
        }

        _stateStack.pop_back();

        if (_stateStack.back() != kExpectArrayStart) {
            return false;
        }

        _stateStack.pop_back();

        return true;
    }

    bool EndObject(SizeType memberCount) {
        if (_stateStack.back() != kExpectKeyOrObjectEnd) {
            return false;
        }

        _stateStack.pop_back();

        if (_stateStack.back() != kExpectObjectStart) {
            return false;
        }

        _stateStack.pop_back();

        return true;
    }

    bool Default() { return false; } // all other events are invalid



    enum State {
        kExpectObjectStart,
        kExpectKeyOrObjectEnd,

        kExpectIntValue,
        kExpectUIntValue,
        kExpectFloatValue,
        kExpectDoubleValue,
        kExpectStringValue,

        kExpectArrayStart,
        kExpectArrayItemOrArrayEnd
    };

    std::vector<State> _stateStack;

    std::string _key;
    std::string _strVal;
    QueryRendererConfig _rendererConfig;

    ScaleObject _scaleObj;
    Value _arrayVal;  // a rapidjson Value object. Using this for an
                     // an all-type-inclusive object.

    bool _isEclecticArray;  // if true, array can hold objects of any type
    Value _array;
    MemoryPoolAllocator<> allocator;

    int *_intRef;
    unsigned int *_uintRef;

    std::vector<std::string> _expectedStrVals;
    std::string *_strRef;
};

} // MapD_Renderer namespace

#endif // RENDER_CONFIG_JSON_HANDLER_H_

#ifndef __REQUEST_H
#define __REQUEST_H

#include "Config.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

namespace ramulator {

class Request {
public:
    // specify request id within the unit of core
    int reqid = -1;

    bool is_first_command = true;
    std::string opcode;
    long dim = -1;
    long value = -1;
    long addr = -1;
    long addr_end = -1;
    long stride = -1;
    enum class Type {
        READ,
        WRITE,
        REFRESH,
        POWERDOWN,
        SELFREFRESH,
        EXTENSION,
        MVE,
        FREE,
        INITIALIZED,
        EVICT_DIRTY,
        EVICT_CLEAN,
        DC_BLOCK,
        MAX
    } type;
    int vid = -1;

    // CB id for microops
    int CB_id = -1;
    int CB_id_dst = -1;

    // The width (number of bits) of the data type
    long data_type = 0;

    // The starting address of the microop for a certain CB
    std::vector<long> addr_starts;

    // The ending address of the microop for a certain CB
    std::vector<long> addr_ends;

    // Individual element addresses for a microop for a certain CB
    vector<int> addr_vec;

    long dst = -1;
    long src1 = -1;
    long src2 = -1;

    bool ready = false;

    long free_reg = 0;

    // specify which core sent this request
    int coreid = -1;

    int dc_blockid = -1;

    // specify which unit in the core sent this request
    enum class UnitID {
        CORE,
        L1,
        L2,
        L3,
        MAX
    } unitid;

    std::string unitid_string;
    std::map<UnitID, string> UNITID_2_STRING = {
        {UnitID::CORE, "CORE"},
        {UnitID::L1, "L1"},
        {UnitID::L2, "L2"},
        {UnitID::L3, "L3"},
        {UnitID::MAX, "MAX"},
    };

    long arrive = -1;
    long depart = -1;
    function<void(Request &)> callback; // call back with more info

    // Minimum SIMD element ID of the microop
    int min_eid = -1;

    // Maximum SIMD element ID of the microop
    int max_eid = -1;

    bool *vector_mask;

    // For MVE loads and stores
    Request(std::string &opcode, long dst, long src1, long src2, long addr, long addr_end, std::vector<long> addr_starts, std::vector<long> addr_ends, long data_type, long stride, bool ready, function<void(Request &)> callback, int coreid = MAX_CORE_ID, UnitID unitid = UnitID::MAX)
        : opcode(opcode), addr(addr), addr_end(addr_end), stride(stride), type(Type::MVE), data_type(data_type), addr_starts(addr_starts), addr_ends(addr_ends), dst(dst), src1(src1), src2(src2), ready(ready), coreid(coreid), unitid(unitid), callback(callback) {
        // assert(coreid == 0);
        unitid_string = UNITID_2_STRING[unitid];
    }

    // For MVE Computations
    Request(std::string &opcode, long dst, long src1, long src2, long data_type, bool ready, function<void(Request &)> callback, int coreid = MAX_CORE_ID, UnitID unitid = UnitID::MAX)
        : opcode(opcode), type(Type::MVE), data_type(data_type), dst(dst), src1(src1), src2(src2), ready(ready), coreid(coreid), unitid(unitid), callback(callback) {
        // assert(coreid == 0);
        unitid_string = UNITID_2_STRING[unitid];
    }

    // For MVE Configurations
    Request(std::string &opcode, long dim, long value, bool ready, function<void(Request &)> callback, int coreid = MAX_CORE_ID, UnitID unitid = UnitID::MAX)
        : opcode(opcode), dim(dim), value(value), type(Type::MVE), ready(ready), coreid(coreid), unitid(unitid), callback(callback) {
        // assert(coreid == 0);
        unitid_string = UNITID_2_STRING[unitid];
    }

    // For Bubbles
    Request(long addr, Type type, int coreid = MAX_CORE_ID, UnitID unitid = UnitID::MAX)
        : addr(addr), type(type), data_type(8), ready(true), coreid(coreid), unitid(unitid), callback([](Request &req) {}) {
        // assert(coreid == 0);
        unitid_string = UNITID_2_STRING[unitid];
    }

    // For CPU Loads and Stores in MVE mode
    Request(long addr, Type type, bool ready, function<void(Request &)> callback, int coreid = MAX_CORE_ID, UnitID unitid = UnitID::MAX)
        : addr(addr), addr_end(addr + 7), type(type), data_type(8), ready(ready), coreid(coreid), unitid(unitid), callback(callback) {
        // assert(coreid == 0);
        addr_starts.push_back(addr);
        addr_ends.push_back(addr + 7);
        unitid_string = UNITID_2_STRING[unitid];
    }

    // For CPU Loads and Stores in other modes
    Request(long addr, Type type, function<void(Request &)> callback, int coreid = MAX_CORE_ID, UnitID unitid = UnitID::MAX)
        : addr(addr), addr_end(addr + 7), type(type), data_type(8), ready(true), coreid(coreid), unitid(unitid), callback(callback) {
        // assert(coreid == 0);
        addr_starts.push_back(addr);
        addr_ends.push_back(addr + 7);
        unitid_string = UNITID_2_STRING[unitid];
    }

    Request(vector<int> &addr_vec, Type type, function<void(Request &)> callback, int coreid = MAX_CORE_ID, UnitID unitid = UnitID::MAX)
        : type(type), addr_vec(addr_vec), ready(true), coreid(coreid), unitid(unitid), callback(callback) {
        // assert(coreid == 0);
        unitid_string = UNITID_2_STRING[unitid];
    }

    Request(int coreid = MAX_CORE_ID, UnitID unitid = UnitID::MAX)
        : ready(true), coreid(coreid), unitid(unitid) {
        // assert(coreid == 0);
        unitid_string = UNITID_2_STRING[unitid];
        opcode = "NONE";
        addr = -1;
        addr_end = -1;
        stride = -1;
        type = Type::MAX;
        data_type = -1;
        addr_starts = std::vector<long>();
        addr_ends = std::vector<long>();
    }

    friend bool operator==(const Request &req1, const Request &req2) {
        if ((req1.coreid == req2.coreid) && (req1.unitid == req2.unitid) && (req1.reqid == req2.reqid) && //(req1.vid == req2.vid) && (req1.vid_dst == req2.vid_dst) &&
            (req1.CB_id == req2.CB_id) && (req1.CB_id_dst == req2.CB_id_dst)) {
            assert(req1.data_type == req2.data_type);
            assert(req1.type == req2.type);
            assert(req1.opcode == req2.opcode);
            assert(req1.dim == req2.dim);
            assert(req1.value == req2.value);
            return true;
        }
        return false;
    }

    friend bool operator<(const Request &req1, const Request &req2) {
        if (req1.coreid < req2.coreid) {
            return true;
        } else if (req1.coreid == req2.coreid) {
            if (req1.unitid < req2.unitid) {
                return true;
            } else if (req1.unitid == req2.unitid) {
                if (req1.reqid < req2.reqid) {
                    return true;
                }
            }
        }
        return false;
    }

    const char *c_str() const {
        std::stringstream req_stream;
        char *name = new char[1024];

        if (type == Type::MVE) {
            if (value != -1) {
                // it's a config instr
                req_stream << "Request[Core(" << coreid << "), Unit(" << unitid_string << "), ID(" << reqid << ")][type(MVE), opcode(" << opcode << "), dim(" << dim << "), value(" << value << ")]";
            } else if (addr != -1) {
                // it's a load or store instruction
                req_stream << "Request[Core(" << coreid << "), Unit(" << unitid_string << "), ID(" << reqid << ")][type(MVE), opcode(" << opcode;
                if (dst != -1) {
                    req_stream << "), dst(" << dst << "), src1(" << src1 << "), src2(" << src2;
                }
                req_stream << "), data(" << data_type << "), addr(0x" << std::hex << addr << " - 0x" << addr_end << "), starts-ends(";
                int addr_to_print = addr_starts.size() > 8 ? 8 : addr_starts.size();
                for (int idx = 0; idx < addr_to_print; idx++) {
                    req_stream << "0x" << addr_starts[idx] << "-0x" << addr_ends[idx];
                    if (idx != addr_starts.size() - 1) {
                        req_stream << ", ";
                    }
                }
                if (addr_starts.size() > 8) {
                    req_stream << "...";
                }
                req_stream << std::dec << "), Stride(" << stride << ")"; //, VID(" << vid << ")";
                if (CB_id != -1) {
                    req_stream << ", SID(" << CB_id << "), min_eid(" << min_eid << "), max_eid(" << max_eid << ")";
                }
                req_stream << "]";
            } else {
                // it's a compute instruction
                req_stream << "Request[Core(" << coreid << "), Unit(" << unitid_string << "), ID(" << reqid << ")][type(MVE), opcode(" << opcode;
                if (dst != -1) {
                    req_stream << "), dst(" << dst << "), src1(" << src1 << "), src2(" << src2;
                }
                req_stream << ")";
                if (CB_id != -1) {
                    req_stream << ", SID(" << CB_id << "), min_eid(" << min_eid << "), max_eid(" << max_eid << ")";
                }
                req_stream << "]";
            }
        } else if (type == Type::READ) {
            req_stream << "Request[Core(" << coreid << "), Unit(" << unitid_string << "), ID(" << reqid << ")][type(READ), addr(0x" << std::hex << addr << ")]";
        } else if (type == Type::WRITE) {
            req_stream << "Request[Core(" << coreid << "), Unit(" << unitid_string << "), ID(" << reqid << ")][type(WRITE), addr(0x" << std::hex << addr << ")]";
        } else if (type == Type::INITIALIZED) {
            req_stream << "Request[Core(" << coreid << "), Unit(" << unitid_string << "), ID(" << reqid << ")][type(INITIALIZED)]";
        } else if (type == Type::MAX) {
            req_stream << "Request[Core(" << coreid << "), Unit(" << unitid_string << "), ID(" << reqid << ")][type(BUBBLE)]";
        } else if (type == Type::EVICT_CLEAN) {
            req_stream << "Request[Core(" << coreid << "), Unit(" << unitid_string << "), ID(" << reqid << ")][type(EVICT_CLEAN)], addr(0x" << std::hex << addr << ")]";
        } else if (type == Type::EVICT_DIRTY) {
            req_stream << "Request[Core(" << coreid << "), Unit(" << unitid_string << "), ID(" << reqid << ")][type(EVICT_DIRTY)], addr(0x" << std::hex << addr << ")]";
        } else if (type == Type::DC_BLOCK) {
            req_stream << "Request[Core(" << coreid << "), Unit(" << unitid_string << "), ID(" << reqid << ")][type(DC_BLOCK)], block(" << addr << ")]";
        } else {
            assert(false);
        }

        strcpy(name, req_stream.str().c_str());
        return name;
    }
};

} /*namespace ramulator*/

#endif /*__REQUEST_H*/

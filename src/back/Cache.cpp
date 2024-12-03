#include "Cache.h"
#include "Config.h"
#include <cassert>
#include <cstdio>
#include <utility>

#ifndef DEBUG_CACHE
#define debug(...)
#else
#define debug(...)                                   \
    do {                                             \
        printf("\033[36m[DEBUG] %s ", __FUNCTION__); \
        printf(__VA_ARGS__);                         \
        printf("\033[0m\n");                         \
    } while (0)
#endif

namespace ramulator {

Cache::Cache(int size, int assoc, int block_size,
             int mshr_entry_num, float access_energy, Level level,
             std::shared_ptr<CacheSystem> cachesys, int MVE_CB_num, int core_id)
    : level(level), cachesys(cachesys), higher_cache(0), lower_cache(nullptr), core_id(core_id), size(size), assoc(assoc), block_size(block_size), mshr_entry_num(mshr_entry_num), access_energy(access_energy), MVE_CB_num(MVE_CB_num) {

    debug("level %d size %d assoc %d block_size %d\n",
          int(level), size, assoc, block_size);

    if (level == Level::L1) {
        level_string = "L1";
    } else if (level == Level::L2) {
        level_string = "L2";
    } else if (level == Level::L3) {
        level_string = "L3";
    }

    is_first_level = (level == cachesys->first_level);
    is_last_level = (level == cachesys->last_level);

    // Check size, block size and assoc are 2^N
    assert((size & (size - 1)) == 0);
    assert((block_size & (block_size - 1)) == 0);
    assert((assoc & (assoc - 1)) == 0);
    assert(size >= block_size);

    // Initialize cache configuration
    block_num = size / (block_size * assoc);
    index_mask = block_num - 1;
    index_offset = calc_log2(block_size);
    tag_offset = calc_log2(block_num) + index_offset;

    for (int i = 0; i < MVE_CB_num; i++) {
        last_MVE_instruction_compute_clk[i] = -1;
        last_MVE_instruction_computed[i] = false;
        last_MVE_instruction_sent[i] = false;
    }

    CB_PER_V = MVE_CB_num;
    DC_reg = 1;
    VL_reg[0] = MVE_CB_num * 256;
    VL_reg[1] = VL_reg[2] = VL_reg[3] = 1;
    VC_reg = 1;
    LS_reg[0] = LS_reg[1] = LS_reg[2] = LS_reg[3] = 0;
    SS_reg[0] = SS_reg[1] = SS_reg[2] = SS_reg[3] = 0;
    VM_reg[0] = vector<bool>(MVE_CB_num * 256);
    fill(VM_reg[0].begin(), VM_reg[0].end(), true);
    VM_reg[1] = vector<bool>(1);
    VM_reg[1][0] = true;
    VM_reg[2] = vector<bool>(1);
    VM_reg[2][0] = true;
    VM_reg[3] = vector<bool>(1);
    VM_reg[3][0] = true;

    debug("index_offset %d", index_offset);
    debug("index_mask 0x%x", index_mask);
    debug("tag_offset %d", tag_offset);

    init_intrinsic_latency();

    // regStats
    cache_read_miss.name(level_string + string("_cache_read_miss"))
        .desc("cache read miss count")
        .precision(0);

    cache_write_miss.name(level_string + string("_cache_write_miss"))
        .desc("cache write miss count")
        .precision(0);

    cache_total_miss.name(level_string + string("_cache_total_miss"))
        .desc("cache total miss count")
        .precision(0);

    cache_eviction.name(level_string + string("_cache_eviction"))
        .desc("number of evict from this level to lower level")
        .precision(0);

    cache_read_access.name(level_string + string("_cache_read_access"))
        .desc("cache read access count")
        .precision(0);

    cache_write_access.name(level_string + string("_cache_write_access"))
        .desc("cache write access count")
        .precision(0);

    cache_total_access.name(level_string + string("_cache_total_access"))
        .desc("cache total access count")
        .precision(0);

    cache_mshr_hit.name(level_string + string("_cache_mshr_hit"))
        .desc("cache mshr hit count")
        .precision(0);
    cache_mshr_unavailable.name(level_string + string("_cache_mshr_unavailable"))
        .desc("cache mshr not available count")
        .precision(0);
    cache_set_unavailable.name(level_string + string("_cache_set_unavailable"))
        .desc("cache set not available")
        .precision(0);
    cache_access_energy.name(level_string + string("_cache_access_energy"))
        .desc("cache access energy in pJ")
        .precision(0);

    MVE_host_device_total_cycles.name(level_string + string("_MVE_host_device_total_cycles"))
        .desc("total cycles at which cache MVE instruction queue is empty")
        .precision(0);
    MVE_move_stall_total_cycles.name(level_string + string("_MVE_move_stall_total_cycles"))
        .desc("total cycles at which cache MVE is stalled because of move instructions")
        .precision(0);
    MVE_compute_total_cycles.name(level_string + string("_MVE_compute_total_cycles"))
        .desc("total cycles at which cache MVE doing computation or W/R")
        .precision(0);
    MVE_memory_total_cycles.name(level_string + string("_MVE_memory_total_cycles"))
        .desc("total cycles at which cache MVE waiting for mem requests")
        .precision(0);

    MVE_host_device_cycles = new ScalarStat[MVE_CB_num];
    MVE_move_stall_cycles = new ScalarStat[MVE_CB_num];
    MVE_compute_cycles = new ScalarStat[MVE_CB_num];
    MVE_memory_cycles = new ScalarStat[MVE_CB_num];
    for (int sid = 0; sid < MVE_CB_num; sid++) {
        MVE_host_device_cycles[sid].name(level_string + string("_MVE_host_device_cycles[") + to_string(sid) + string("]")).desc("cache MVE instruction queue is empty").precision(0);
        MVE_move_stall_cycles[sid].name(level_string + string("_MVE_move_stall_cycles[") + to_string(sid) + string("]")).desc("cache MVE is stalled for move instruction").precision(0);
        MVE_compute_cycles[sid].name(level_string + string("_MVE_compute_cycles[") + to_string(sid) + string("]")).desc("cache MVE doing computation or W/R").precision(0);
        MVE_memory_cycles[sid].name(level_string + string("_MVE_memory_cycles[") + to_string(sid) + string("]")).desc("cache MVE waiting for mem requests").precision(0);
    }

    MVE_compute_total_energy.name(level_string + string("_MVE_compute_total_energy"))
        .desc("total cache MVE compute energy in pJ")
        .precision(0);
    MVE_compute_comp_total_energy.name(level_string + string("_MVE_compute_comp_total_energy"))
        .desc("total cache MVE compute energy [compute part] in pJ")
        .precision(0);
    MVE_compute_rdwr_total_energy.name(level_string + string("_MVE_compute_rdwr_total_energy"))
        .desc("total cache MVE compute energy [read/write part] in pJ")
        .precision(0);

    MVE_compute_energy = new ScalarStat[MVE_CB_num];
    MVE_compute_comp_energy = new ScalarStat[MVE_CB_num];
    MVE_compute_rdwr_energy = new ScalarStat[MVE_CB_num];

    for (int sid = 0; sid < MVE_CB_num; sid++) {
        MVE_compute_energy[sid].name(level_string + string("_MVE_compute_energy[") + to_string(sid) + string("]")).desc("cache MVE compute energy in pJ").precision(0);
        MVE_compute_comp_energy[sid].name(level_string + string("_MVE_compute_comp_energy[") + to_string(sid) + string("]")).desc("cache MVE compute energy [compute part] in pJ").precision(0);
        MVE_compute_rdwr_energy[sid].name(level_string + string("_MVE_compute_rdwr_energy[") + to_string(sid) + string("]")).desc("cache MVE compute energy [read/write part] in pJ").precision(0);
    }
}

void Cache::init_intrinsic_latency() {
    std::string line, intrinsic, rd_wr_latency, compute_latency;

    fstream file("/home/arkhadem/MVE/ramulator/data/MVE_intrinsics_latency.csv", ios::in);
    if (file.is_open()) {

        // reading and ignoring header file
        std::getline(file, line);

        // reading all intrinsics
        while (std::getline(file, line)) {

            stringstream str(line);

            std::getline(str, intrinsic, ',');
            std::getline(str, rd_wr_latency, ',');
            std::getline(str, compute_latency, ',');

            MVE_ACCESS_DELAY[intrinsic.c_str()] = atoi(rd_wr_latency.c_str());
            MVE_COMPUTE_DELAY[intrinsic.c_str()] = atoi(compute_latency.c_str());
        }
    } else {
        printf("Error: could not find /home/arkhadem/MVE/ramulator/data/MVE_intrinsics_latency.csv\n");
    }
    file.close();

    file.open("/home/arkhadem/MVE/ramulator/data/dc_intrinsics_latency.csv", ios::in);
    if (file.is_open()) {

        // reading and ignoring header file
        std::getline(file, line);

        // reading all intrinsics
        while (std::getline(file, line)) {

            stringstream str(line);

            std::getline(str, intrinsic, ',');
            std::getline(str, rd_wr_latency, ',');
            std::getline(str, compute_latency, ',');

            DC_ACCESS_DELAY[intrinsic.c_str()] = atoi(rd_wr_latency.c_str());
            DC_COMPUTE_DELAY[intrinsic.c_str()] = atoi(compute_latency.c_str());
        }
    } else {
        printf("Error: could not find /home/arkhadem/MVE/ramulator/data/dc_intrinsics_latency.csv\n");
    }
    file.close();
}

int Cache::vid_to_sid(int vid, int base = 0) {
    if (VL_reg[0] <= 256) {
        return vid / V_PER_CB + base;
    } else {
        return vid * CB_PER_V + base;
    }
}

bool Cache::check_full_queue(Request req) {
    // if (req.vid == -1) {
    // assert(req.opcode.find("move") == string::npos);
    for (int vid = 0; vid < VC_reg; vid += V_PER_CB) {
        for (int sid_offset = 0; sid_offset < CB_PER_V; sid_offset++) {
            if (MVE_instruction_queue[vid_to_sid(vid, sid_offset)].size() >= MAX_MVE_QUEUE_SIZE) {
                return false;
            }
        }
    }
    // } else {
    //     for (int sid_offset = 0; sid_offset < CB_PER_V; sid_offset++) {
    //         if (MVE_instruction_queue[vid_to_sid(req.vid, sid_offset)].size() >= MAX_MVE_QUEUE_SIZE) {
    //             return false;
    //         }
    //     }
    // }

    // if (req.opcode.find("move") != string::npos) {
    //     for (int sid_offset = 0; sid_offset < CB_PER_V; sid_offset++) {
    //         if (MVE_instruction_queue[vid_to_sid(req.vid_dst, sid_offset)].size() >= MAX_MVE_QUEUE_SIZE) {
    //             return false;
    //         }
    //     }
    // }
    return true;
}

int Cache::stride_evaluator(long rstride, bool load) {
    switch (rstride) {
    case 3:
        if (load)
            return LS_reg[0];
        else
            return SS_reg[0];
        break;
    case 2:
        return 1;
        break;
    case 1:
        return rstride;
        break;
    case 0:
        return rstride;
        break;
    default:
        assert(false);
    }
}

bool Cache::vector_masked(int vid) {
    int temp_vid = vid;
    for (int dim = 0; dim < 4; dim++) {
        if (VM_reg[dim][temp_vid % (int)VL_reg[dim]] == false) {
            hint("VID %d is masked b/c VM_reg[%d][%d] = false", vid, dim, temp_vid % (int)VL_reg[dim]);
            return true;
        }
        temp_vid /= VL_reg[dim];
    }
    return false;
}

void Cache::instrinsic_decoder(Request req) {

    assert(MVE_vop_to_num_sop.count(req) == 0);

    // if (req.opcode.find("move") != string::npos) {
    //     // move cannot have vid = -1
    //     assert(req.vid >= 0);
    //     MVE_vop_to_num_sop[req] = CB_PER_V;
    //     if ((req.vid / V_PER_CB) != (req.vid_dst / V_PER_CB)) {
    //         MVE_vop_to_num_sop[req] *= 2;
    //     }
    //     // For each SA of the vector
    //     for (int sid_offset = 0; sid_offset < CB_PER_V; sid_offset++) {
    //         req.sid = vid_to_sid(req.vid, sid_offset);
    //         req.sid_dst = vid_to_sid(req.vid_dst, sid_offset);
    //         hint("%s set for start in %d clock cycles\n", req.c_str(), latency_each[int(level)]);
    //         MVE_instruction_queue[req.sid].push_back(make_pair(cachesys->clk + latency_each[int(level)], req));
    //         if (req.sid != req.sid_dst) {
    //             MVE_instruction_queue[req.sid_dst].push_back(make_pair(cachesys->clk + latency_each[int(level)], req));
    //         }
    //     }
    // } else

    hint("Decoding %s\n", req.c_str());

    if ((req.opcode.find("load") != string::npos) || (req.opcode.find("store") != string::npos)) {

        std::vector<long> addr_starts = req.addr_starts;
        std::vector<long> addr_ends = req.addr_ends;

        int stride = stride_evaluator(req.stride, (req.opcode.find("load") != string::npos));

        // if (req.vid == -1) {
        MVE_vop_to_num_sop[req] = ((VC_reg - 1) / V_PER_CB) + 1;
        MVE_vop_to_num_sop[req] *= CB_PER_V;

        // For each vector
        for (int vid_base = 0; vid_base < VC_reg; vid_base += V_PER_CB) {
            // For each SA of the vector
            for (int sid_offset = 0; sid_offset < CB_PER_V; sid_offset++) {
                req.addr_starts.clear();
                req.addr_ends.clear();
                req.sid = vid_to_sid(vid_base, sid_offset);
                req.stride = stride;
                req.min_eid = sid_offset * 256;
                req.max_eid = (req.min_eid + 256) < VL_reg[0] ? (req.min_eid + 256) : VL_reg[0];

                // For each vector of the SA
                int remaining_vectors = (V_PER_CB < (VC_reg - vid_base)) ? (V_PER_CB) : (VC_reg - vid_base);
                for (int vid_offset = 0; vid_offset < remaining_vectors; vid_offset++) {
                    // VID shows which address pair should be used
                    int vid = vid_base + vid_offset;

                    if (vector_masked(vid)) {
                        // This vector is masked
                        continue;
                    }

                    // It's an ordinary load or store
                    long addr = addr_starts[vid] + (long)(std::ceil((float)(sid_offset * 256 * req.data_type * stride / 8)));
                    long addr_end;
                    if (stride == 0)
                        addr_end = min(((long)(std::ceil((float)(req.data_type / 8))) + addr - 1), addr_ends[vid]);
                    else
                        addr_end = min(((long)(std::ceil((float)(256 * req.data_type * stride / 8))) + addr - 1), addr_ends[vid]);

                    req.addr_starts.push_back(addr);
                    req.addr_ends.push_back(addr_end);

                    if (vid_offset == 0) {
                        req.addr = req.addr_starts[0];
                        req.addr_end = req.addr_ends[req.addr_ends.size() - 1];
                    }
                }

                if (req.addr_starts.size() == 0) {
                    // All vectors of this SRAM array are masked
                    MVE_vop_to_num_sop[req]--;
                    continue;
                }

                // Schedule the instruction
                hint("%s set for start in %d clock cycles\n", req.c_str(), latency_each[int(level)]);
                MVE_instruction_queue[req.sid].push_back(make_pair(cachesys->clk + latency_each[int(level)], req));
            }

            assert(MVE_vop_to_num_sop[req] >= 0);

            if (MVE_vop_to_num_sop[req] == 0) {
                hint("20- Calling back %s to core\n", req.c_str());
                req.callback(req);
            }
        }
        // } else {
        //     assert((addr_starts.size() == 1) && (addr_ends.size() == 1));
        //     MVE_vop_to_num_sop[req] = CB_PER_V;

        //     // For each SA of the vector
        //     for (int sid_offset = 0; sid_offset < CB_PER_V; sid_offset++) {
        //         req.sid = vid_to_sid(req.vid, sid_offset);

        //         if ((req.opcode.find("load1") == string::npos) && (req.opcode.find("store1") == string::npos)) {
        //             // It's an ordinary load or store
        //             req.addr_starts.clear();
        //             req.addr_ends.clear();
        //             long addr = addr_starts[0] + (long)(std::ceil((float)(sid_offset * 256 * req.data_type / 8)));
        //             long addr_end = min(((long)(std::ceil((float)(256 * req.data_type / 8))) + addr - 1), addr_ends[0]);
        //             req.addr_starts.push_back(addr);
        //             req.addr_ends.push_back(addr_end);
        //             req.addr = req.addr_starts[0];
        //         }

        //         // Schedule the instruction
        //         hint("%s set for start in %d clock cycles\n", req.c_str(), latency_each[int(level)]);
        //         MVE_instruction_queue[req.sid].push_back(make_pair(cachesys->clk + latency_each[int(level)], req));
        //     }
        // }
    } else {
        // if (req.vid == -1) {
        MVE_vop_to_num_sop[req] = ((VC_reg - 1) / V_PER_CB) + 1;
        MVE_vop_to_num_sop[req] *= CB_PER_V;

        // For each vector
        for (int vid_base = 0; vid_base < VC_reg; vid_base += V_PER_CB) {
            // For each SA of the vector
            for (int sid_offset = 0; sid_offset < CB_PER_V; sid_offset++) {
                req.sid = vid_to_sid(vid_base, sid_offset);

                // For each vector of the SA
                bool SA_mased = true;
                int remaining_vectors = (V_PER_CB < (VC_reg - vid_base)) ? (V_PER_CB) : (VC_reg - vid_base);
                for (int vid_offset = 0; vid_offset < remaining_vectors; vid_offset++) {
                    // VID shows which address pair should be used
                    int vid = vid_base + vid_offset;

                    if (vector_masked(vid)) {
                        // This vector is masked
                        continue;
                    }

                    SA_mased = false;
                    break;
                }

                if (SA_mased) {
                    // All vectors of this SRAM array are masked
                    MVE_vop_to_num_sop[req]--;
                    continue;
                }

                // Schedule the instruction
                hint("%s set for start in %d clock cycles\n", req.c_str(), latency_each[int(level)]);
                MVE_instruction_queue[req.sid].push_back(make_pair(cachesys->clk + latency_each[int(level)], req));
            }
        }

        if (MVE_vop_to_num_sop[req] == 0) {
            hint("21- Calling back %s to core\n", req.c_str());
            req.callback(req);
        }
        // } else {
        //     MVE_vop_to_num_sop[req] = CB_PER_V;
        //     // For each SA of the vector
        //     for (int sid_offset = 0; sid_offset < CB_PER_V; sid_offset++) {
        //         req.sid = vid_to_sid(req.vid, sid_offset);
        //         // Schedule the instruction
        //         hint("%s set for start in %d clock cycles\n", req.c_str(), latency_each[int(level)]);
        //         MVE_instruction_queue[req.sid].push_back(make_pair(cachesys->clk + latency_each[int(level)], req));
        //     }
        // }
    }
}

void Cache::random_access_decoder(Request req) {
    // Accessing memory to load random load/store addresses
    assert(MVE_random_to_mem_ops.count(req) == 0);
    MVE_random_to_mem_ops[req] = std::vector<long>();
    long lower_cache_line = align(req.addr);
    long upper_cache_line = align(req.addr + (VL_reg[DC_reg - 1] * 8) - 1);
    int access_needed = (long)(std::ceil((float)(upper_cache_line - lower_cache_line) / (float)(block_size))) + 1;
    long req_addr = lower_cache_line;
    Request::Type req_type = Request::Type::READ;
    for (int i = 0; i < access_needed; i++) {
        int req_coreid = req.coreid;
        Request::UnitID req_unitid = (Request::UnitID)(level);
        Request mem_req(req_addr, req_type, processor_callback, req_coreid, req_unitid);
        mem_req.reqid = last_id;
        last_id++;
        MVE_random_to_mem_ops[req].push_back(req_addr);
        req_addr += block_size;

        // send it
        hint("10- %s sending %s to %s\n", level_string.c_str(), mem_req.c_str(), level_string.c_str());
        if (send(mem_req) == false) {
            self_retry_list.push_back(mem_req);
        }
    }
}

bool Cache::MVE_controller(Request req) {
    if (req.opcode.find("_set_") != string::npos) {
        // it's a config MVE instruction

        if (req.opcode.find("stride") != string::npos) {
            assert(req.dim < DC_reg);
            if (req.opcode.find("load") != string::npos) {
                LS_reg[req.dim] = req.value;
            } else if (req.opcode.find("store") != string::npos) {
                SS_reg[req.dim] = req.value;
            } else {
                assert(false);
            }
        } else if (req.opcode.find("dim") != string::npos) {
            if (req.opcode.find("length") != string::npos) {
                assert(req.dim < DC_reg);
                VL_reg[req.dim] = req.value;
                if (VL_reg[0] * VL_reg[1] * VL_reg[2] * VL_reg[3] > (256 * MVE_CB_num)) {
                    printf("Error: VL_reg[0](%ld) * VL_reg[1](%ld) * VL_reg[2](%ld) * VL_reg[3](%ld) > (256 * MVE_CB_num(%d))", VL_reg[0], VL_reg[1], VL_reg[2], VL_reg[3], MVE_CB_num);
                    exit(-1);
                }
                VC_reg = VL_reg[1] * VL_reg[2] * VL_reg[3];
                VM_reg[req.dim].clear();
                VM_reg[req.dim] = std::vector<bool>(req.value);
                fill(VM_reg[req.dim].begin(), VM_reg[req.dim].end(), true);
                if (req.dim == 0) {
                    if (req.value <= 256) {
                        V_PER_CB = (255 / req.value) + 1;
                        CB_PER_V = 1;
                    } else {
                        CB_PER_V = ((req.value - 1) / 256) + 1;
                        V_PER_CB = 1;
                    }
                }
            } else if (req.opcode.find("count") != string::npos) {
                DC_reg = req.dim;
            } else {
                assert(false);
            }
        } else if (req.opcode.find("element") != string::npos) {
            assert(req.dim < DC_reg);
            bool val = true;
            if (req.opcode.find("unset") != string::npos) {
                val = false;
            }
            if (req.opcode.find("all") != string::npos) {
                for (int element = 0; element < VL_reg[req.dim]; element++) {
                    VM_reg[req.dim][element] = val;
                }
            } else if (req.opcode.find("only") != string::npos) {
                for (int element = 0; element < VL_reg[req.dim]; element++) {
                    VM_reg[req.dim][element] = !val;
                }
                VM_reg[req.dim][req.value] = val;
            } else if (req.opcode.find("active") != string::npos) {
                VM_reg[req.dim][req.value] = val;
            } else {
                assert(false);
            }
        } else {
            assert(false);
        }
        hint("DC_reg: %ld, LS_reg: [%ld, %ld, %ld, %ld], SS_reg: [%ld, %ld, %ld, %ld], VL_reg: [%ld, %ld, %ld, %ld], VC_reg: %ld, V_PER_CB: %d, CB_PER_V: %d\n", DC_reg, LS_reg[0], LS_reg[1], LS_reg[2], LS_reg[3], SS_reg[0], SS_reg[1], SS_reg[2], SS_reg[3], VL_reg[0], VL_reg[1], VL_reg[2], VL_reg[3], VC_reg, V_PER_CB, CB_PER_V);
    } else {

        if (((((VC_reg * CB_PER_V) + 1) / V_PER_CB) - 1) > MVE_CB_num) {
            printf("Error: ((((VC_reg(%ld) * CB_PER_V(%d)) + 1) / V_PER_CB(%d)) - 1) (%ld) > MVE_CB_num(%d)\n", VC_reg, CB_PER_V, V_PER_CB, ((((VC_reg * CB_PER_V) + 1) / V_PER_CB) - 1), MVE_CB_num);
            exit(-1);
        }

        if (check_full_queue(req) == false) {
            hint("SA Queue is full, returning False for %s\n", req.c_str());
            return false;
        }

        if ((req.opcode.find("loadr") != string::npos) || (req.opcode.find("storer") != string::npos)) {
            random_access_decoder(req);
        } else {
            instrinsic_decoder(req);
        }
    }
    return true;
}

bool Cache::send(Request req) {
    if (req.type == Request::Type::MVE) {
        debug("level %s received %s", level_string.c_str(), req.c_str());

        return MVE_controller(req);
    }
    // else if (req.type == Request::Type::EVICT_DIRTY) {
    //     evictline(req.addr, true);
    //     return true;
    // } else if (req.type == Request::Type::EVICT_CLEAN) {
    //     evictline(req.addr, false);
    //     return true;
    // }

    // req.addr = align(req.addr);
    // req.addr_end = align(req.addr) + 7;
    // req.addr_starts[0] = req.addr;
    // req.addr_ends[0] = req.addr_end;

    debug("level %s received %s, index %d, tag %ld\n",
          level_string.c_str(), req.c_str(), get_index(req.addr),
          get_tag(req.addr));

    cache_total_access++;
    if (req.type == Request::Type::WRITE) {
        cache_write_access++;
    } else {
        assert(req.type == Request::Type::READ);
        cache_read_access++;
    }
    // If there isn't a set, create it.
    auto &lines = get_lines(req.addr);
    std::list<Line>::iterator line;

    if (is_hit(lines, req.addr, &line)) {

        bool dirty = line->dirty || (req.type == Request::Type::WRITE);
        long invalidate_time = 0;

        if (req.unitid == (Request::UnitID)(level)) {
            // If it is comming from the same level of the cache, it is produced by a MVE intrinsic

            if (higher_cache.size() != 0) {
                // Make sure it is not L1

                for (auto hc : higher_cache) {
                    std::pair<long, bool> result;
                    if (hc->invalidate(req.addr, result) == false)
                        return false;
                    invalidate_time = max(invalidate_time, result.first + (result.second ? latency_each[int(level)] : 0));
                    dirty = dirty || result.second;

                    if (result.second) {
                        hint("invalidated (%s) from %s due to MVE access\n", req.c_str(), level_string.c_str());
                    }
                }
            }
        }

        lines.push_back(Line(req.addr, get_tag(req.addr), false, dirty));
        hint("1- %s: Pushed back addr 0x%lx, idx 0x%x, and tag 0x%lx\n", level_string.c_str(), req.addr, get_index(req.addr), get_tag(req.addr));

#ifdef DEBUG
        auto cache_line = cache_lines.find(get_index(req.addr));
        if (cache_line == cache_lines.end()) {
            hint("No line found for the aforementioned idx and tag");
        } else {
            auto liinees = cache_line->second;
            auto l = liinees.begin();
            while (l != liinees.end()) {
                hint("Line addr: 0x%lx tag: 0x%lx\n", l->addr, l->tag);
                ++l;
            }
        }
#endif

        lines.erase(line);

#ifdef DEBUG
        hint("1- %s: Erased addr 0x%lx, idx 0x%x, and tag 0x%lx\n", level_string.c_str(), req.addr, get_index(req.addr), get_tag(req.addr));
        if (cache_line == cache_lines.end()) {
            hint("No line found for the aforementioned idx and tag");
        } else {
            auto liinees = cache_line->second;
            auto l = liinees.begin();
            while (l != liinees.end()) {
                hint("Line addr: 0x%lx tag: 0x%lx\n", l->addr, l->tag);
                ++l;
            }
        }
#endif

        cachesys->hit_list.push_back(make_pair(cachesys->clk + latency_each[int(level)] + invalidate_time, req));

        debug("hit, update timestamp %ld", cachesys->clk);
        debug("hit finish time %ld", cachesys->clk + latency_each[int(level)]);

        // Reading/writing block_size bytes from cache for a hit
        cache_access_energy += access_energy;

        return true;
    } else {
        debug("%s missed @level %d", req.c_str(), (level));
        cache_total_miss++;
        if (req.type == Request::Type::WRITE) {
            cache_write_miss++;
        } else {
            assert(req.type == Request::Type::READ);
            cache_read_miss++;
        }

        // The dirty bit will be set if this is a write request and @L1
        bool dirty = (req.type == Request::Type::WRITE);

        // Modify the type of the request to lower level
        if (req.type == Request::Type::WRITE) {
            req.type = Request::Type::READ;
        }

        // Look it up in MSHR entries
        assert(req.type == Request::Type::READ);
        auto mshr = hit_mshr(req.addr);
        if (mshr != mshr_entries.end()) {
            debug("%s hitted mshr", req.c_str());
            cache_mshr_hit++;
            mshr->second->dirty = (dirty || mshr->second->dirty);
            return true;
        }

        // All requests come to this stage will be READ, so they
        // should be recorded in MSHR entries.
        if (mshr_entries.size() == mshr_entry_num) {
            // When no MSHR entries available, the miss request
            // is stalling.
            cache_mshr_unavailable++;
            debug("no mshr entry available");
            return false;
        }

        // Check whether there is a line available
        if (all_sets_locked(lines)) {
            cache_set_unavailable++;
            return false;
        }

        auto newline = allocate_line(lines, req.addr);
        if (newline == lines.end()) {
            return false;
        }

        newline->dirty = dirty;

        // Add to MSHR entries
        mshr_entries.push_back(make_pair(req.addr, newline));
        hint("pair(0x%lx, line(0x%lx, %d, %d, 0x%lx)) added to mshr entries at level %s\n", req.addr, newline->addr, newline->dirty, newline->lock, newline->tag, level_string.c_str());

        // Send the request to next level;
        std::pair<long, Request> time_req = make_pair(cachesys->clk + latency_each[int(level)], req);
        if (!is_last_level) {
            retry_list.push_back(time_req);
        } else {
            hint("11- %s sending %s to cachesystem waitlist\n", level_string.c_str(), req.c_str());
            cachesys->wait_list.push_back(time_req);
        }
        return true;
    }
}

void Cache::evictline(long addr, bool dirty) {

    auto it = cache_lines.find(get_index(addr));
    assert(it != cache_lines.end()); // check inclusive cache
    auto &lines = it->second;
    auto line = find_if(lines.begin(), lines.end(),
                        [addr, this](Line l) { return (l.tag == get_tag(addr)); });
    if (line == lines.end()) {
        printf("%s: addr 0x%lx, idx 0x%x, and tag 0x%lx not found!\n", level_string.c_str(), addr, get_index(addr), get_tag(addr));
        auto cache_line = cache_lines.find(get_index(addr));
        if (cache_line == cache_lines.end()) {
            hint("No line found for the aforementioned idx and tag");
        } else {
            auto liinees = cache_line->second;
            auto l = liinees.begin();
            while (l != liinees.end()) {
                hint("Line addr: 0x%lx tag: 0x%lx\n", l->addr, l->tag);
                ++l;
            }
        }
        exit(-2);
    } else {
        hint("line for addr 0x%lx evicted from %s\n", addr, level_string.c_str());
    }
    assert(line != lines.end());
    // Update LRU queue. The dirty bit will be set if the dirty
    // bit inherited from higher level(s) is set.
    lines.push_back(Line(addr, get_tag(addr), false, dirty || line->dirty));

#ifdef DEBUG
    hint("2- %s: Pushed back addr 0x%lx, idx 0x%x, and tag 0x%lx\n", level_string.c_str(), addr, get_index(addr), get_tag(addr));
    auto cache_line = cache_lines.find(get_index(addr));
    if (cache_line == cache_lines.end()) {
        hint("No line found for the aforementioned idx and tag");
    } else {
        auto liinees = cache_line->second;
        auto l = liinees.begin();
        while (l != liinees.end()) {
            hint("Line addr: 0x%lx tag: 0x%lx\n", l->addr, l->tag);
            ++l;
        }
    }
#endif

    lines.erase(line);

#ifdef DEBUG
    hint("2- %s: Erased addr 0x%lx, idx 0x%x, and tag 0x%lx\n", level_string.c_str(), line->addr, get_index(line->addr), get_tag(line->addr));
    if (cache_line == cache_lines.end()) {
        hint("No line found for the aforementioned idx and tag");
    } else {
        auto liinees = cache_line->second;
        auto l = liinees.begin();
        while (l != liinees.end()) {
            hint("Line addr: 0x%lx tag: 0x%lx\n", l->addr, l->tag);
            ++l;
        }
    }
#endif
}

bool Cache::invalidate(long addr, std::pair<long, bool> &result) {
    long delay = latency_each[int(level)];
    bool dirty = false;

    auto &lines = get_lines(addr);
    if (lines.size() == 0) {
        // The line of this address doesn't exist.
        result = make_pair(0, false);
        return true;
    }
    auto line = find_if(lines.begin(), lines.end(),
                        [addr, this](Line l) { return (l.tag == get_tag(addr)); });

    // If the line is in this level cache, then erase it from the buffer.
    if (line != lines.end()) {
        if (line->lock) {
            return false;
        }
        debug("invalidate 0x%lx @ level %d", addr, int(level));
        dirty = line->dirty;
        lines.erase(line);

#ifdef DEBUG
        hint("3- %s: Erased addr 0x%lx, idx 0x%x, and tag 0x%lx\n", level_string.c_str(), line->addr, get_index(line->addr), line->tag);
        auto cache_line = cache_lines.find(get_index(addr));
        if (cache_line == cache_lines.end()) {
            hint("No line found for the aforementioned idx and tag");
        } else {
            auto liinees = cache_line->second;
            auto l = liinees.begin();
            while (l != liinees.end()) {
                hint("Line addr: 0x%lx tag: 0x%lx\n", l->addr, l->tag);
                ++l;
            }
        }
#endif

    } else {
        // If it's not in current level, then no need to go up.
        result = make_pair(delay, false);
        return true;
    }

    if (higher_cache.size()) {
        long max_delay = delay;
        for (auto hc : higher_cache) {
            std::pair<long, bool> result;
            if (hc->invalidate(addr, result) == false)
                return false;
            if (result.second) {
                max_delay = max(max_delay, delay + result.first * 2);
            } else {
                max_delay = max(max_delay, delay + result.first);
            }
            dirty = dirty || result.second;
        }
        delay = max_delay;
    }

    result = make_pair(delay, dirty);
    return true;
}

bool Cache::evict(std::list<Line> *lines, std::list<Line>::iterator victim) {
    debug("level %d miss evict victim 0x%lx", int(level), victim->addr);
    // Before anything, check if this address exists in lower cache
    if (!is_last_level) {
        if (lower_cache->exists_addr(victim->addr) == false) {
            debug("line is not received by the lower cahce yet, returning false by evict\n");
            return false;
        }
    }

    cache_eviction++;

    long addr = victim->addr;
    long invalidate_time = 0;
    bool dirty = victim->dirty;

    // First invalidate the victim line in higher level.
    if (higher_cache.size()) {
        for (auto hc : higher_cache) {
            std::pair<long, bool> result;
            assert(hc->invalidate(addr, result));
            invalidate_time = max(invalidate_time,
                                  result.first + (result.second ? latency_each[int(level)] : 0));
            dirty = dirty || result.second || victim->dirty;
        }
    }

    debug("invalidate delay: %ld, dirty: %s", invalidate_time, dirty ? "true" : "false");

    if (!is_last_level) {
        // not LLC eviction
        assert(lower_cache != nullptr);
        // Request::Type type = dirty ? Request::Type::EVICT_DIRTY : Request::Type::EVICT_CLEAN;
        // Request evict_req(addr, type, 0, (Request::UnitID)(level));
        // evict_req.reqid = last_id;
        // last_id++;
        // std::pair<long, Request> time_req = make_pair(cachesys->clk + latency_each[int(level)], evict_req);
        // retry_list.push_back(time_req);
        // hint("%s set request %s for execution\n", level_string.c_str(), evict_req.c_str());
        lower_cache->evictline(addr, dirty);
    } else {
        // LLC eviction
        if (dirty) {
            // Request write_req(addr, Request::Type::WRITE, MAX_CORE_ID, (Request::UnitID)(level));
            Request write_req(addr, Request::Type::WRITE, 0, (Request::UnitID)(level));
            cachesys->wait_list.push_back(make_pair(
                cachesys->clk + invalidate_time + latency_each[int(level)],
                write_req));

            // Reading block_size for eviction from LLC and writing to memory
            cache_access_energy += access_energy;

            debug("inject one write request to memory system "
                  "addr 0x%lx, invalidate time %ld, issue time %ld",
                  write_req.addr, invalidate_time,
                  cachesys->clk + invalidate_time + latency_each[int(level)]);
        }
    }

    lines->erase(victim);

#ifdef DEBUG
    hint("4- %s: Erased addr 0x%lx, idx 0x%x, and tag 0x%lx\n", level_string.c_str(), victim->addr, get_index(victim->addr), victim->tag);
    auto cache_line = cache_lines.find(get_index(victim->addr));
    if (cache_line == cache_lines.end()) {
        hint("No line found for the aforementioned idx and tag");
    } else {
        auto liinees = cache_line->second;
        auto l = liinees.begin();
        while (l != liinees.end()) {
            hint("Line addr: 0x%lx tag: 0x%lx\n", l->addr, l->tag);
            ++l;
        }
    }
#endif

    return true;
}

std::list<Cache::Line>::iterator Cache::allocate_line(std::list<Line> &lines, long addr) {
    // See if an eviction is needed
    if (need_eviction(lines, addr)) {
        // Get victim.
        // The first one might still be locked due to reorder in MC
        auto victim = find_if(lines.begin(), lines.end(),
                              [this](Line line) {
                                  bool check = !line.lock;
                                  if (!is_first_level) {
                                      for (auto hc : higher_cache) {
                                          if (!check) {
                                              return check;
                                          }
                                          check = check && hc->check_unlock(line.addr);
                                      }
                                  }
                                  return check;
                              });
        if (victim == lines.end()) {
            return victim; // doesn't exist a line that's already unlocked in each level
        }
        assert(victim != lines.end());
        if (evict(&lines, victim) == false)
            return lines.end();
    }

    // Allocate newline, with lock bit on and dirty bit off
    lines.push_back(Line(addr, get_tag(addr)));
    hint("3- %s: Pushed back addr 0x%lx, idx 0x%x, and tag 0x%lx\n", level_string.c_str(), addr, get_index(addr), get_tag(addr));

#ifdef DEBUG
    auto cache_line = cache_lines.find(get_index(addr));
    if (cache_line == cache_lines.end()) {
        hint("No line found for the aforementioned idx and tag");
    } else {
        auto liinees = cache_line->second;
        auto l = liinees.begin();
        while (l != liinees.end()) {
            hint("Line addr: 0x%lx tag: 0x%lx\n", l->addr, l->tag);
            ++l;
        }
    }
#endif

    hint("%s addr 0x%lx pushed to lines!\n", level_string.c_str(), addr);
    auto last_element = lines.end();
    --last_element;

    // Writing block_size bytes to cache as a result of a miss
    cache_access_energy += access_energy;

    return last_element;
}

bool Cache::is_hit(std::list<Line> &lines, long addr, std::list<Line>::iterator *pos_ptr) {
    auto pos = find_if(lines.begin(), lines.end(), [addr, this](Line l) { return (l.tag == get_tag(addr)); });
    *pos_ptr = pos;
    if (pos == lines.end()) {
        return false;
    }
    return !pos->lock;
}

void Cache::concatlower(Cache *lower) {
    lower_cache = lower;
    assert(lower != nullptr);
    lower->higher_cache.push_back(this);
};

bool Cache::need_eviction(const std::list<Line> &lines, long addr) {
    if (find_if(lines.begin(), lines.end(),
                [addr, this](Line l) { return (get_tag(addr) == l.tag); }) != lines.end()) {
        // Due to MSHR, the program can't reach here. Just for checking
        assert(false);
    } else {
        if (lines.size() < assoc) {
            return false;
        } else {
            return true;
        }
    }
}

void Cache::callback(Request &req) {
    debug("level %d", int(level));
    hint("%s received in %s\n", req.c_str(), level_string.c_str());

    // Remove related MSHR entries
    auto MSHR_it = find_if(mshr_entries.begin(), mshr_entries.end(),
                           [&req, this](std::pair<long, std::list<Line>::iterator> mshr_entry) {
                               return (align(mshr_entry.first) == align(req.addr));
                           });

    if (MSHR_it != mshr_entries.end()) {
        assert(MSHR_it->second != NULL);
        MSHR_it->second->lock = false;
        hint("pair(0x%lx, line(0x%lx, %d, %d, 0x%lx)) removed from mshr entries at level %s\n", MSHR_it->first, MSHR_it->second->addr, MSHR_it->second->dirty, MSHR_it->second->lock, MSHR_it->second->tag, level_string.c_str());
        mshr_entries.erase(MSHR_it);
    } else {
        hint("NO MSHR entry removed at %s\n", level_string.c_str());
    }

    // Remove corresponding random load/store addresses
    auto random_it = MVE_random_to_mem_ops.begin();
    while (random_it != MVE_random_to_mem_ops.end()) {
        auto mem_it = random_it->second.begin();
        while (mem_it != random_it->second.end()) {
            if (align(req.addr) == align(*mem_it)) {
                hint("2- %s: %s calls back for %s, %lu instructions remained\n", level_string.c_str(), req.c_str(), random_it->first.c_str(), random_it->second.size() - 1);

                // Remove this instruction from awaiting random accesses
                mem_it = random_it->second.erase(mem_it);
            } else {
                ++mem_it;
            }
        }
        if (random_it->second.size() == 0) {
            hint("%s: decoding %s\n", level_string.c_str(), random_it->first.c_str());
            instrinsic_decoder(random_it->first);
            random_it = MVE_random_to_mem_ops.erase(random_it);
        } else {
            ++random_it;
        }
    }

    // Remove corresponding MVE instructions
    for (int sid = 0; sid < MVE_CB_num; sid++) {

        int hit = 0;

        // Check if the SA has sent the memory operations
        if ((last_MVE_instruction_computed[sid] == true) && (last_MVE_instruction_sent[sid] == true)) {
            Request MVE_req = MVE_compute_queue[sid][0].second;

            // Check all start-end address pairs
            for (int MVE_idx = 0; MVE_idx < MVE_req.addr_starts.size(); MVE_idx++) {

                // If the address has overlap with the start-end pair
                if ((align(req.addr) >= align(MVE_req.addr_starts[MVE_idx])) && (align(req.addr) <= align(MVE_req.addr_ends[MVE_idx]))) {

                    // Check if this memory access has been ocurred because of this MVE instruction
                    auto iter = MVE_op_to_mem_ops[sid][MVE_req].begin();
                    while (iter != MVE_op_to_mem_ops[sid][MVE_req].end()) {
                        if (iter->second == false) {
                            ++iter;
                            continue;
                        }
                        if (align(req.addr) == align(iter->first.addr)) {
                            hint("1- %s: %s calls back for %s, %lu instructions remained\n", level_string.c_str(), req.c_str(), MVE_req.c_str(), MVE_op_to_mem_ops[sid][MVE_req].size() - 1);
                            hit += 1;
                            // Remove this instruction from MVE list
                            iter = MVE_op_to_mem_ops[sid][MVE_req].erase(iter);
                        } else {
                            ++iter;
                        }
                    }
                }
            }
            if (MVE_op_to_mem_ops[sid][MVE_req].size() == 0) {
                op_trace << cachesys->clk << " " << sid << " F " << MVE_req.opcode << endl;
                hint("18- %s: calling back %s\n", level_string.c_str(), MVE_req.c_str());
                callbacker(MVE_req);
                MVE_op_to_mem_ops[sid].erase(MVE_req);
                MVE_compute_queue[sid].erase(MVE_compute_queue[sid].begin());
                last_MVE_instruction_compute_clk[sid] = -1;
                last_MVE_instruction_computed[sid] = false;
                last_MVE_instruction_sent[sid] = false;
            } else if (hit != 0) {
                // send the first non-sent addresses
                for (int mem_idx = 0; mem_idx < MVE_op_to_mem_ops[sid][MVE_req].size(); mem_idx++) {
                    if (MVE_op_to_mem_ops[sid][MVE_req][mem_idx].second == true)
                        continue;
                    hint("15- %s sending %s to %s\n", level_string.c_str(), MVE_op_to_mem_ops[sid][MVE_req][mem_idx].first.c_str(), level_string.c_str());
                    if (send(MVE_op_to_mem_ops[sid][MVE_req][mem_idx].first) == false) {
                        self_retry_list.push_back(MVE_op_to_mem_ops[sid][MVE_req][mem_idx].first);
                    }
                    assert(MVE_op_to_mem_ops[sid][MVE_req][mem_idx].second == false);
                    MVE_op_to_mem_ops[sid][MVE_req][mem_idx].second = true;
                    hit--;
                    if (hit == 0)
                        break;
                }
            }
        }
    }

    if (higher_cache.size()) {
        for (auto hc : higher_cache) {
            hc->callback(req);
        }
    }
}

void Cache::callbacker(Request &req) {
    assert(MVE_vop_to_num_sop.count(req) == 1);
    assert(MVE_vop_to_num_sop[req] > 0);
    MVE_vop_to_num_sop[req] -= 1;
    if (MVE_vop_to_num_sop[req] == 0) {
        hint("19- Calling back %s to core\n", req.c_str());
        req.callback(req);
    }
}

bool addr_exists(std::vector<std::pair<Request, bool>> req_vector, long addr) {
    for (int i = 0; i < req_vector.size(); i++) {
        if (req_vector[i].first.addr == addr)
            return true;
    }
    return false;
}

void Cache::tick() {

    if (!is_last_level)
        if (!lower_cache->is_last_level)
            lower_cache->tick();

    auto it = retry_list.begin();
    while (it != retry_list.end()) {
        if (cachesys->clk >= it->first) {
            hint("9- %s sending %s to %s\n", level_string.c_str(), it->second.c_str(), lower_cache->level_string.c_str());
            if (lower_cache->send(it->second)) {
                it = retry_list.erase(it);
            } else {
                ++it;
            }
        } else {
            ++it;
        }
    }

    while (self_retry_list.size() != 0) {
        hint("12- %s sending %s to %s\n", level_string.c_str(), self_retry_list[0].c_str(), level_string.c_str());
        if (send(self_retry_list[0])) {
            self_retry_list.erase(self_retry_list.begin());
        } else {
            break;
        }
    }

    // Instruction received by cache, sent to MVE core queue
    for (int sid = 0; sid < MVE_CB_num; sid++) {
        while ((MVE_instruction_queue[sid].size() > 0) && (cachesys->clk >= MVE_instruction_queue[sid][0].first)) {
            Request req = MVE_instruction_queue[sid][0].second;
            long compute_delay, access_delay, bitlines;
            if (req.opcode.find("_dc_") != string::npos) {
                assert(req.opcode.find("_pc3_") == string::npos);
                compute_delay = DC_COMPUTE_DELAY[req.opcode];
                access_delay = DC_ACCESS_DELAY[req.opcode];
                bitlines = 4;
            } else {
                assert(req.opcode.find("_pc3_") != string::npos);
                compute_delay = MVE_COMPUTE_DELAY[req.opcode];
                access_delay = MVE_ACCESS_DELAY[req.opcode];
                bitlines = 1;
            }
            // hint("%s %d %d\n", req.opcode.c_str(), compute_delay, access_delay);
            if ((compute_delay + access_delay) == 0) {
                MVE_instruction_queue[sid].erase(MVE_instruction_queue[sid].begin());
                continue;
            }
            assert((compute_delay + access_delay) > 0);
            hint("%s set for compute in %ld clock cycles\n", req.c_str(), compute_delay + access_delay);
            MVE_instruction_queue[sid].erase(MVE_instruction_queue[sid].begin());
            MVE_compute_queue[sid].push_back(make_pair(compute_delay + access_delay, req));
            MVE_compute_total_energy += ((float)compute_delay * 15.4 + (float)access_delay * 8.6) * (float)bitlines;
            MVE_compute_comp_total_energy += ((float)compute_delay * 15.4) * (float)bitlines;
            MVE_compute_rdwr_total_energy += ((float)access_delay * 8.6) * (float)bitlines;
            MVE_compute_energy[sid] += ((float)compute_delay * 15.4 + (float)access_delay * 8.6) * (float)bitlines;
            MVE_compute_comp_energy[sid] += ((float)compute_delay * 15.4) * (float)bitlines;
            MVE_compute_rdwr_energy[sid] += ((float)access_delay * 8.6) * (float)bitlines;
        }
    }

    // Instruction at the head of the MVE queue is computed
    // If it's a store or load it is unpacked
    // Otherwise, it is called back
    for (int sid = 0; sid < MVE_CB_num; sid++) {

        // Check if there is any instruction ready for completion
        if ((last_MVE_instruction_computed[sid] == true) && (last_MVE_instruction_sent[sid] == false)) {

            // There must be an instruction going on
            assert(MVE_compute_queue[sid].size() != 0);

            // Check if the instruction is done
            if (cachesys->clk - last_MVE_instruction_compute_clk[sid] >= MVE_compute_queue[sid].at(0).first) {
                Request req = MVE_compute_queue[sid].at(0).second;

                if ((req.opcode.find("load") != string::npos) || (req.opcode.find("store") != string::npos)) {
                    // If it's a load or store, make new queries and send to this cache level's queue

                    hint("Unpacking loads/stores for: %s\n", req.c_str());

                    assert(MVE_op_to_mem_ops[sid].count(req) == 0);
                    MVE_op_to_mem_ops[sid][req] = std::vector<std::pair<Request, bool>>();
                    MVE_op_to_mem_ops[sid][req].clear();
                    for (int idx = 0; idx < req.addr_starts.size(); idx++) {
                        if (req.addr_starts[idx] == 0) {
                            hint("13- %s ignored one zero-addressed memory access for %s\n", level_string.c_str(), req.c_str());
                            continue;
                        }
                        // if (req.stride * req.data_type / 8 <= block_size) {
                        //     // Load/store all cache lines
                        //     long lower_cache_line = align(req.addr_starts[idx]);
                        //     long upper_cache_line = align(req.addr_ends[idx]);
                        //     int access_needed = (long)(std::ceil((float)(upper_cache_line - lower_cache_line) / (float)(block_size))) + 1;
                        //     hint("13- %s unpacking %d instructions for %s\n", level_string.c_str(), access_needed, req.c_str());
                        //     assert(access_needed > 0);
                        //     long req_addr = lower_cache_line;
                        //     for (int i = 0; i < access_needed; i++) {
                        //         // Check if this memory access has happenned before
                        //         if (addr_exists(MVE_op_to_mem_ops[sid][req], req_addr)) {
                        //             hint("14- %s NOT sending 0x%lx to %s\n", level_string.c_str(), req_addr, level_string.c_str());
                        //         } else {
                        //             // make the request
                        //             Request::Type req_type = (req.opcode.find("load") != string::npos) ? Request::Type::READ : Request::Type::WRITE;
                        //             int req_coreid = req.coreid;
                        //             Request::UnitID req_unitid = (Request::UnitID)(level);
                        //             Request mem_req(req_addr, req_type, processor_callback, req_coreid, req_unitid);
                        //             mem_req.reqid = last_id;
                        //             last_id++;
                        //             MVE_op_to_mem_ops[sid][req].push_back(std::pair<Request, bool>(mem_req, false));
                        //         }
                        //         req_addr += block_size;
                        //     }
                        // } else {
                        long addr = req.addr_starts[idx];
                        for (int i = req.min_eid; i < req.max_eid; i++) {
                            if ((addr < req.addr_starts[idx]) || (addr >= req.addr_ends[idx])) {
                                printf("Error: (addr(0x%lx) < req.addr_starts[%d](0x%lx)) || (addr(0x%lx) >= req.addr_ends[%d](0x%lx))", addr, i, req.addr_starts[idx], addr, i, req.addr_ends[idx]);
                                exit(-1);
                            }
                            assert(VM_reg[0].size() > i);
                            if (VM_reg[0][i]) {
                                if (addr_exists(MVE_op_to_mem_ops[sid][req], align(addr))) {
                                    hint("14- %s NOT sending 0x%lx to %s\n", level_string.c_str(), align(addr), level_string.c_str());
                                } else {
                                    // make the request
                                    Request::Type req_type = (req.opcode.find("load") != string::npos) ? Request::Type::READ : Request::Type::WRITE;
                                    int req_coreid = req.coreid;
                                    Request::UnitID req_unitid = (Request::UnitID)(level);
                                    Request mem_req(align(addr), req_type, processor_callback, req_coreid, req_unitid);
                                    mem_req.reqid = last_id;
                                    last_id++;
                                    MVE_op_to_mem_ops[sid][req].push_back(std::pair<Request, bool>(mem_req, false));
                                    hint("unpacked: %s\n", mem_req.c_str());
                                }
                            }
                            addr += (req.stride * req.data_type / 8);
                        }
                        // }
                    }

                    last_MVE_instruction_sent[sid] = true;

                    if (MVE_op_to_mem_ops[sid][req].size() == 0) {
                        op_trace << cachesys->clk << " " << sid << " F " << req.opcode << endl;
                        hint("17- %s: calling back %s\n", level_string.c_str(), req.c_str());
                        callbacker(req);
                        MVE_op_to_mem_ops[sid].erase(req);
                        MVE_compute_queue[sid].erase(MVE_compute_queue[sid].begin());
                        last_MVE_instruction_compute_clk[sid] = -1;
                        last_MVE_instruction_computed[sid] = false;
                        last_MVE_instruction_sent[sid] = false;
                    } else {
                        // send the first address
                        for (int mem_idx = 0; mem_idx < (mshr_entry_num / MVE_CB_num); mem_idx++) {
                            if (mem_idx >= MVE_op_to_mem_ops[sid][req].size())
                                break;
                            hint("15- %s sending %s to %s\n", level_string.c_str(), MVE_op_to_mem_ops[sid][req][mem_idx].first.c_str(), level_string.c_str());
                            if (send(MVE_op_to_mem_ops[sid][req][mem_idx].first) == false) {
                                self_retry_list.push_back(MVE_op_to_mem_ops[sid][req][mem_idx].first);
                            }
                            MVE_op_to_mem_ops[sid][req][mem_idx].second = true;
                        }
                    }

                } else {
                    // Otherwise, we are ready to send the call back
                    op_trace << cachesys->clk << " " << sid << " F " << req.opcode << endl;

                    hint("16- %s instruction %s completed\n", level_string.c_str(), req.c_str());
                    callbacker(req);
                    MVE_compute_queue[sid].erase(MVE_compute_queue[sid].begin());
                    last_MVE_instruction_compute_clk[sid] = -1;
                    last_MVE_instruction_computed[sid] = false;
                    last_MVE_instruction_sent[sid] = false;
                }
            }
        }
    }

    // Instruction at the head of the MVE queue gets ready for compute
    for (int sid = 0; sid < MVE_CB_num; sid++) {
        // Check if there is any instructions ready to be computed
        if (last_MVE_instruction_computed[sid] == false) {

            // The last instruction must not be sent
            assert(last_MVE_instruction_sent[sid] == false);

            if (MVE_compute_queue[sid].size() != 0) {
                // A new instruction must be computed
                if (MVE_compute_queue[sid].at(0).second.opcode.find("move") != string::npos) {

                    // we should wait for the dst SA as well
                    int sid_src = MVE_compute_queue[sid].at(0).second.sid;
                    int sid_dst = MVE_compute_queue[sid].at(0).second.sid_dst;
                    assert(sid_dst != -1);

                    if (sid_src == sid_dst) {
                        op_trace << cachesys->clk << " " << sid << " S " << MVE_compute_queue[sid].at(0).second.opcode << endl;
                        hint("Computing %s at %ld, %zu instructions in compute queue\n", MVE_compute_queue[sid].at(0).second.c_str(), cachesys->clk, MVE_compute_queue[sid].size());
                        last_MVE_instruction_compute_clk[sid] = cachesys->clk;
                        last_MVE_instruction_computed[sid] = true;
                    } else if (sid != sid_dst) {
                        // only src SA checks for the dst, not vice versa

                        // Check if there is any instructions ready to be computed in the dst queue
                        if (last_MVE_instruction_computed[sid_dst] == false) {

                            // The last instruction must not be sent
                            assert(last_MVE_instruction_sent[sid_dst] == false);

                            if (MVE_compute_queue[sid_dst].size() != 0) {

                                // Check if these are the same instructions
                                if (MVE_compute_queue[sid].at(0).second == MVE_compute_queue[sid_dst].at(0).second) {

                                    // Compute both
                                    op_trace << cachesys->clk << " " << sid << " S " << MVE_compute_queue[sid].at(0).second.opcode << endl;
                                    op_trace << cachesys->clk << " " << sid_dst << " S " << MVE_compute_queue[sid].at(0).second.opcode << endl;
                                    hint("Computing %s at %ld, %zu instructions in compute queue\n", MVE_compute_queue[sid].at(0).second.c_str(), cachesys->clk, MVE_compute_queue[sid].size());
                                    hint("Computing %s at %ld, %zu instructions in compute queue\n", MVE_compute_queue[sid_dst].at(0).second.c_str(), cachesys->clk, MVE_compute_queue[sid_dst].size());
                                    last_MVE_instruction_compute_clk[sid] = cachesys->clk;
                                    last_MVE_instruction_compute_clk[sid_dst] = cachesys->clk;
                                    last_MVE_instruction_computed[sid] = true;
                                    last_MVE_instruction_computed[sid_dst] = true;
                                }
                            }
                        }

                        if (last_MVE_instruction_computed[sid] == false) {
                            hint("%s is waiting for dst SA\n", MVE_compute_queue[sid].at(0).second.c_str());
                        }
                    }

                } else {
                    op_trace << cachesys->clk << " " << sid << " S " << MVE_compute_queue[sid].at(0).second.opcode << endl;
                    hint("Computing %s at %ld, %zu instructions in compute queue\n", MVE_compute_queue[sid].at(0).second.c_str(), cachesys->clk, MVE_compute_queue[sid].size());
                    last_MVE_instruction_compute_clk[sid] = cachesys->clk;
                    last_MVE_instruction_computed[sid] = true;
                }
            }
        }
    }

    for (int sid = 0; sid < MVE_CB_num; sid++) {
        if ((last_MVE_instruction_computed[sid] == false) && (last_MVE_instruction_sent[sid] == false)) {

            if (MVE_compute_queue[sid].size() == 0) {
                // There is no instruction ready for execute
                hint("%s MVE %d HOST_DEVICE...\n", level_string.c_str(), sid);
                MVE_host_device_total_cycles++;
                MVE_host_device_cycles[sid]++;
            } else {
                // Instruction at top must be move and stalled by dst SA
                hint("%s MVE %d MOVE...\n", level_string.c_str(), sid);
                assert(MVE_compute_queue[sid].at(0).second.opcode.find("move") != string::npos);
                MVE_move_stall_total_cycles++;
                MVE_move_stall_cycles[sid]++;
            }
        }

        if ((last_MVE_instruction_computed[sid] == true) && (last_MVE_instruction_sent[sid] == false)) {
            // An instruction is being computed
            hint("%s MVE %d COMPUTE...\n", level_string.c_str(), sid);
            MVE_compute_total_cycles++;
            MVE_compute_cycles[sid]++;
        }

        if ((last_MVE_instruction_computed[sid] == true) && (last_MVE_instruction_sent[sid] == true)) {
            // An instruction is computed and sent, waiting for memory instructions to be called back
            hint("%s MVE %d MEMORY...\n", level_string.c_str(), sid);
            MVE_memory_total_cycles++;
            MVE_memory_cycles[sid]++;
        }
    }
}

bool Cache::finished() {
    if (mshr_entries.size() != 0)
        return false;
    if (retry_list.size() != 0)
        return false;
    if (self_retry_list.size() != 0)
        return false;

    if (MVE_random_to_mem_ops.size() != 0)
        return false;

    for (int sid = 0; sid < MVE_CB_num; sid++) {
        if (MVE_op_to_mem_ops[sid].size() != 0)
            return false;

        if (MVE_instruction_queue[sid].size() != 0)
            return false;

        if (MVE_compute_queue[sid].size() != 0)
            return false;
    }
    return true;
}

void Cache::reset_state() {
    if (level == Level::L3)
        hint("Cache %s state reset\n", level_string.c_str());
    else
        hint("Core %d's Cache %s state reset\n", core_id, level_string.c_str());

    cache_read_miss = 0;
    cache_write_miss = 0;
    cache_total_miss = 0;
    cache_eviction = 0;
    cache_read_access = 0;
    cache_write_access = 0;
    cache_total_access = 0;
    cache_mshr_hit = 0;
    cache_mshr_unavailable = 0;
    cache_set_unavailable = 0;
    cache_access_energy = 0;
    MVE_host_device_total_cycles = 0;
    MVE_move_stall_total_cycles = 0;
    MVE_compute_total_cycles = 0;
    MVE_memory_total_cycles = 0;
    MVE_compute_total_energy = 0;
    MVE_compute_comp_total_energy = 0;
    MVE_compute_rdwr_total_energy = 0;
    assert(MVE_random_to_mem_ops.size() == 0);
    for (int sid = 0; sid < MVE_CB_num; sid++) {
        MVE_host_device_cycles[sid] = 0;
        MVE_move_stall_cycles[sid] = 0;
        MVE_compute_cycles[sid] = 0;
        MVE_memory_cycles[sid] = 0;
        MVE_compute_energy[sid] = 0;
        MVE_compute_comp_energy[sid] = 0;
        MVE_compute_rdwr_energy[sid] = 0;
        last_MVE_instruction_computed[sid] = false;
        last_MVE_instruction_sent[sid] = false;
        assert(MVE_op_to_mem_ops[sid].size() == 0);

        for (int i = 0; i < MVE_instruction_queue[sid].size(); i++) {
            printf("ERROR: %s remained in MVE instruction queue %s\n", MVE_instruction_queue[sid].at(0).second.c_str(), level_string.c_str());
        }
        assert(MVE_instruction_queue[sid].size() == 0);
        for (int i = 0; i < MVE_compute_queue[sid].size(); i++) {
            printf("ERROR: %s remained in MVE compute queue %s\n", MVE_compute_queue[sid].at(0).second.c_str(), level_string.c_str());
        }
        assert(MVE_compute_queue[sid].size() == 0);
    }

    last_id = 0;

    for (int idx = 0; idx < mshr_entries.size(); idx++) {
        auto first = mshr_entries[idx].first;
        auto second = mshr_entries[idx].second;
        printf("Error: pair(0x%lx, line(0x%lx, %d, %d, 0x%lx)) remained in mshr entries at level %s\n", first, second->addr, second->dirty, second->lock, second->tag, level_string.c_str());
    }
    assert(mshr_entries.size() == 0);
    assert(retry_list.size() == 0);
    assert(self_retry_list.size() == 0);

    CB_PER_V = MVE_CB_num;
    DC_reg = 1;
    VL_reg[0] = MVE_CB_num * 256;
    VL_reg[1] = VL_reg[2] = VL_reg[3] = 1;
    VC_reg = 1;
    LS_reg[0] = LS_reg[1] = LS_reg[2] = LS_reg[3] = 0;
    SS_reg[0] = SS_reg[1] = SS_reg[2] = SS_reg[3] = 0;
    VM_reg[0] = vector<bool>(MVE_CB_num * 256);
    fill(VM_reg[0].begin(), VM_reg[0].end(), true);
    VM_reg[1] = vector<bool>(1);
    VM_reg[1][0] = true;
    VM_reg[2] = vector<bool>(1);
    VM_reg[2][0] = true;
    VM_reg[3] = vector<bool>(1);
    VM_reg[3][0] = true;
}

void CacheSystem::tick() {
    debug("clk %ld", clk);

    ++clk;

    // Sends ready waiting request to memory
    auto it = wait_list.begin();
    while (it != wait_list.end() && clk >= it->first) {
        if (!send_memory(it->second)) {

            debug("failed sending %s to memory", (it->second).c_str());

            ++it;
        } else {

            debug("complete req: %s", (it->second).c_str());

            it = wait_list.erase(it);
        }
    }

    // hit request callback
    it = hit_list.begin();
    while (it != hit_list.end()) {
        if (clk >= it->first) {
            it->second.callback(it->second);

            debug("finish hit: %s", (it->second).c_str());

            it = hit_list.erase(it);
        } else {
            ++it;
        }
    }
}

void CacheSystem::reset_state() {
    hint("CacheSystem state reset\n");
    clk = 0;
    auto it = wait_list.begin();
    while (it != wait_list.end()) {
        debug("%s\n", it->second.c_str());
        ++it;
    }
    assert(wait_list.size() == 0);

    it = hit_list.begin();
    while (it != hit_list.end()) {
        debug("%s\n", it->second.c_str());
        ++it;
    }
    assert(hit_list.size() == 0);
}

bool CacheSystem::finished() {
    return (wait_list.size() == 0) && (hit_list.size() == 0);
}

} // namespace ramulator

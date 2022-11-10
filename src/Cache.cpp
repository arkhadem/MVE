#include "Cache.h"
#include "Config.h"
#include "Processor.h"
#include "Request.h"
#include <cassert>
#include <cstdio>
#include <memory>
#include <stdlib.h>
#include <time.h>
#include <utility>
#include <vector>

namespace ramulator {

Cache::Cache(int size, int assoc, int block_size,
             int mshr_entry_num, float access_energy, Level level,
             std::shared_ptr<CacheSystem> cachesys, int gpic_core_num, int core_id)
    : level(level), cachesys(cachesys), higher_cache(0), lower_cache(nullptr), core_id(core_id), size(size), assoc(assoc), block_size(block_size), mshr_entry_num(mshr_entry_num), access_energy(access_energy), gpic_core_num(gpic_core_num) {

    hint("level %d size %d assoc %d block_size %d\n",
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

    for (int i = 0; i < gpic_core_num; i++) {
        last_gpic_instruction_compute_clk[i] = -1;
        last_gpic_instruction_computed[i] = false;
        last_gpic_instruction_sent[i] = false;
        SA_masked[i] = false;
    }

    SA_PER_V = gpic_core_num;
    DC_reg = 1;
    VL_reg[0] = gpic_core_num * 256;
    VL_reg[1] = VL_reg[2] = VL_reg[3] = 1;
    VC_reg = 1;
    LS_reg[0] = LS_reg[1] = LS_reg[2] = LS_reg[3] = 0;
    SS_reg[0] = SS_reg[1] = SS_reg[2] = SS_reg[3] = 0;
    VM_reg[0] = vector<bool>(gpic_core_num * 256);
    fill(VM_reg[0].begin(), VM_reg[0].end(), true);
    VM_reg[1] = vector<bool>(1);
    VM_reg[1][0] = true;
    VM_reg[2] = vector<bool>(1);
    VM_reg[2][0] = true;
    VM_reg[3] = vector<bool>(1);
    VM_reg[3][0] = true;

    hint("index_offset %d\n", index_offset);
    hint("index_mask 0x%x\n", index_mask);
    hint("tag_offset %d\n", tag_offset);

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

    GPIC_host_device_total_cycles.name(level_string + string("_GPIC_host_device_total_cycles"))
        .desc("total cycles at which cache GPIC instruction queue is empty")
        .precision(0);
    GPIC_move_stall_total_cycles.name(level_string + string("_GPIC_move_stall_total_cycles"))
        .desc("total cycles at which cache GPIC is stalled because of move instructions")
        .precision(0);
    GPIC_compute_total_cycles.name(level_string + string("_GPIC_compute_total_cycles"))
        .desc("total cycles at which cache GPIC doing computation or W/R")
        .precision(0);
    GPIC_memory_total_cycles.name(level_string + string("_GPIC_memory_total_cycles"))
        .desc("total cycles at which cache GPIC waiting for mem requests")
        .precision(0);

    GPIC_host_device_cycles = new ScalarStat[gpic_core_num];
    GPIC_move_stall_cycles = new ScalarStat[gpic_core_num];
    GPIC_compute_cycles = new ScalarStat[gpic_core_num];
    GPIC_memory_cycles = new ScalarStat[gpic_core_num];
    for (int sid = 0; sid < gpic_core_num; sid++) {
        GPIC_host_device_cycles[sid].name(level_string + string("_GPIC_host_device_cycles[") + to_string(sid) + string("]")).desc("cache GPIC instruction queue is empty").precision(0);
        GPIC_move_stall_cycles[sid].name(level_string + string("_GPIC_move_stall_cycles[") + to_string(sid) + string("]")).desc("cache GPIC is stalled for move instruction").precision(0);
        GPIC_compute_cycles[sid].name(level_string + string("_GPIC_compute_cycles[") + to_string(sid) + string("]")).desc("cache GPIC doing computation or W/R").precision(0);
        GPIC_memory_cycles[sid].name(level_string + string("_GPIC_memory_cycles[") + to_string(sid) + string("]")).desc("cache GPIC waiting for mem requests").precision(0);
    }

    GPIC_compute_total_energy.name(level_string + string("_GPIC_compute_total_energy"))
        .desc("total cache GPIC compute energy in pJ")
        .precision(0);
    GPIC_compute_comp_total_energy.name(level_string + string("_GPIC_compute_comp_total_energy"))
        .desc("total cache GPIC compute energy [compute part] in pJ")
        .precision(0);
    GPIC_compute_rdwr_total_energy.name(level_string + string("_GPIC_compute_rdwr_total_energy"))
        .desc("total cache GPIC compute energy [read/write part] in pJ")
        .precision(0);

    GPIC_compute_energy = new ScalarStat[gpic_core_num];
    GPIC_compute_comp_energy = new ScalarStat[gpic_core_num];
    GPIC_compute_rdwr_energy = new ScalarStat[gpic_core_num];

    for (int sid = 0; sid < gpic_core_num; sid++) {
        GPIC_compute_energy[sid].name(level_string + string("_GPIC_compute_energy[") + to_string(sid) + string("]")).desc("cache GPIC compute energy in pJ").precision(0);
        GPIC_compute_comp_energy[sid].name(level_string + string("_GPIC_compute_comp_energy[") + to_string(sid) + string("]")).desc("cache GPIC compute energy [compute part] in pJ").precision(0);
        GPIC_compute_rdwr_energy[sid].name(level_string + string("_GPIC_compute_rdwr_energy[") + to_string(sid) + string("]")).desc("cache GPIC compute energy [read/write part] in pJ").precision(0);
    }
}

void Cache::init_intrinsic_latency() {
    std::string line, intrinsic, rd_wr_latency, compute_latency;

    fstream file("/home/arkhadem/GPIC/ramulator/data/gpic_intrinsics_latency.csv", ios::in);
    if (file.is_open() == false) {
        file.open("/home/home/ramulator/data/gpic_intrinsics_latency.csv", ios::in);
    }
    if (file.is_open() == false) {
        file.open("/home/arkhadem/ramulator/data/gpic_intrinsics_latency.csv", ios::in);
    }
    if (file.is_open() == false) {
        printf("Error: could not find gpic latency csv file!\n");
        exit(-1);
    }
    if (file.is_open()) {

        // reading and ignoring header file
        std::getline(file, line);

        // reading all intrinsics
        while (std::getline(file, line)) {

            stringstream str(line);

            std::getline(str, intrinsic, ',');
            std::getline(str, rd_wr_latency, ',');
            std::getline(str, compute_latency, ',');

            GPIC_ACCESS_DELAY[intrinsic.c_str()] = atoi(rd_wr_latency.c_str());
            GPIC_COMPUTE_DELAY[intrinsic.c_str()] = atoi(compute_latency.c_str());
        }
    } else {
        printf("Error: could not find /home/arkhadem/GPIC/ramulator/data/gpic_intrinsics_latency.csv\n");
        exit(-1);
    }
    file.close();
}

int Cache::vid_to_sid(int vid, int base = 0) {
    if (VL_reg[0] <= 256) {
        return vid / V_PER_SA + base;
    } else {
        return vid * SA_PER_V + base;
    }
}

bool Cache::check_full_queue(Request req) {
    for (int vid = 0; vid < VC_reg; vid += V_PER_SA) {
        for (int sid_offset = 0; sid_offset < SA_PER_V; sid_offset++) {
            if (gpic_compute_queue[vid_to_sid(vid, sid_offset)].size() >= MAX_GPIC_QUEUE_SIZE) {
                return false;
            }
        }
    }
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
    for (int dim = 1; dim < 4; dim++) {
        if (VM_reg[dim][temp_vid % (int)VL_reg[dim]] == false) {
            hint("VID %d is masked b/c VM_reg[%d][%d] = false\n", vid, dim, temp_vid % (int)VL_reg[dim]);
            return true;
        }
        temp_vid /= VL_reg[dim];
    }
    hint("VID %d is not masked\n", vid);
    return false;
}

void Cache::intrinsic_computer(Request req) {
    long compute_delay, access_delay, bitlines;
    if (req.opcode.find("_dc_") != string::npos) {
        // DC is no longer supported
        assert("false");
    }
    assert(req.opcode.find("_pc3_") != string::npos);
    compute_delay = GPIC_COMPUTE_DELAY[req.opcode];
    access_delay = GPIC_ACCESS_DELAY[req.opcode];
    bitlines = 1;
    // hint("%s %d %d\n", req.opcode.c_str(), compute_delay, access_delay);
    if ((compute_delay + access_delay) == 0) {
        assert(req.opcode.find("cvt") != string::npos);
    }
    hint("%s set for compute in %ld clock cycles\n", req.c_str(), compute_delay + access_delay);
    gpic_compute_queue[req.sid].push_back(make_pair(compute_delay + access_delay, req));
    hint("Compute queue [%d] added size: %d\n", req.sid, (int)gpic_compute_queue[req.sid].size());
    GPIC_compute_total_energy += ((float)compute_delay * 15.4 + (float)access_delay * 8.6) * (float)bitlines;
    GPIC_compute_comp_total_energy += ((float)compute_delay * 15.4) * (float)bitlines;
    GPIC_compute_rdwr_total_energy += ((float)access_delay * 8.6) * (float)bitlines;
    GPIC_compute_energy[req.sid] += ((float)compute_delay * 15.4 + (float)access_delay * 8.6) * (float)bitlines;
    GPIC_compute_comp_energy[req.sid] += ((float)compute_delay * 15.4) * (float)bitlines;
    GPIC_compute_rdwr_energy[req.sid] += ((float)access_delay * 8.6) * (float)bitlines;
    // }
}

void Cache::instrinsic_decoder(Request req) {

    assert(gpic_vop_to_num_sop.count(req) == 0);

    hint("Decoding %s\n", req.c_str());

    req.vector_mask = VM_reg[0];

    if ((req.opcode.find("load") != string::npos) || (req.opcode.find("store") != string::npos)) {

        std::vector<long> addr_starts = req.addr_starts;
        std::vector<long> addr_ends = req.addr_ends;

        int stride = stride_evaluator(req.stride, (req.opcode.find("load") != string::npos));

        gpic_vop_to_num_sop[req] = ((VC_reg - 1) / V_PER_SA) + 1;
        gpic_vop_to_num_sop[req] *= SA_PER_V;

        // For each vector
        for (int vid_base = 0; vid_base < VC_reg; vid_base += V_PER_SA) {
            // For each SA of the vector
            for (int sid_offset = 0; sid_offset < SA_PER_V; sid_offset++) {
                req.addr_starts.clear();
                req.addr_ends.clear();
                req.sid = vid_to_sid(vid_base, sid_offset);

                if (SA_masked[req.sid]) {
                    // All vectors of this SRAM array are masked
                    gpic_vop_to_num_sop[req]--;
                    continue;
                }

                req.stride = stride;
                req.min_vid = sid_offset * 256;
                req.max_vid = (req.min_vid + 256) < VL_reg[0] ? (req.min_vid + 256) : VL_reg[0];

                // For each vector of the SA
                int remaining_vectors = (V_PER_SA < (VC_reg - vid_base)) ? (V_PER_SA) : (VC_reg - vid_base);
                for (int vid_offset = 0; vid_offset < remaining_vectors; vid_offset++) {
                    // VID shows which address pair should be used
                    int vid = vid_base + vid_offset;

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

                // Schedule the instruction
                intrinsic_computer(req);
            }

            assert(gpic_vop_to_num_sop[req] >= 0);

            if (gpic_vop_to_num_sop[req] == 0) {
                hint("20- Calling back %s to core\n", req.c_str());
                req.callback(req);
            }
        }
    } else {
        gpic_vop_to_num_sop[req] = ((VC_reg - 1) / V_PER_SA) + 1;
        gpic_vop_to_num_sop[req] *= SA_PER_V;

        // For each vector
        for (int vid_base = 0; vid_base < VC_reg; vid_base += V_PER_SA) {
            // For each SA of the vector
            for (int sid_offset = 0; sid_offset < SA_PER_V; sid_offset++) {
                req.sid = vid_to_sid(vid_base, sid_offset);

                if (SA_masked[req.sid]) {
                    // All vectors of this SRAM array are masked
                    hint("Request %s masked!\n", req.c_str());
                    gpic_vop_to_num_sop[req]--;
                    continue;
                }

                // Schedule the instruction
                intrinsic_computer(req);
            }
        }

        if (gpic_vop_to_num_sop[req] == 0) {
            hint("21- Calling back %s to core\n", req.c_str());
            req.callback(req);
        }
    }
}

void Cache::random_dict_access_decoder(Request req) {
    // Accessing memory to load random load/store addresses
    assert(gpic_random_dict_to_mem_ops.count(req) == 0);
    gpic_random_dict_to_mem_ops[req] = std::vector<long>();
    long lower_cache_line = align(req.addr);
    long upper_cache_line;
    if (req.opcode.find("dict") != string::npos)
        upper_cache_line = req.addr_end;
    else
        upper_cache_line = align(req.addr + (VL_reg[DC_reg - 1] * 8) - 1);
    int access_needed = ((upper_cache_line - lower_cache_line) / block_size) + 1;
    long req_addr = lower_cache_line;
    Request::Type req_type = Request::Type::READ;
    for (int i = 0; i < access_needed; i++) {
        int req_coreid = req.coreid;
        Request::UnitID req_unitid = (Request::UnitID)(level);
        Request mem_req(req_addr, req_type, true, processor_callback, req_coreid, req_unitid);
        mem_req.reqid = last_id;
        last_id++;
        gpic_random_dict_to_mem_ops[req].push_back(req_addr);
        req_addr += block_size;

        // send it
        hint("10- %s sending %s to %s\n", level_string.c_str(), mem_req.c_str(), level_string.c_str());
        bool should = should_send(mem_req);
        bool sent = false;
        if (should == true) {
            sent = send(mem_req);
        }
        if ((should == false) || (sent == false)) {
            hint("4- should (%d) or sent (%d) is false!\n", should, sent);
            self_retry_list.push_back(mem_req);
        }
    }
    assert(locked == false);
    locked = true;
    hint("locking...\n");
}

bool Cache::memory_controller(Request req) {
    if (req.opcode.find("set_") != string::npos) {
        // it's a config GPIC instruction
        if (req.opcode.find("mask") != string::npos) {
            // do nothing
            assert(true);
        } else if (req.opcode.find("stride") != string::npos) {
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
                if (VL_reg[0] * VL_reg[1] * VL_reg[2] * VL_reg[3] > (256 * gpic_core_num)) {
                    printf("Error: VL_reg[0](%ld) * VL_reg[1](%ld) * VL_reg[2](%ld) * VL_reg[3](%ld) > (256 * gpic_core_num(%d))", VL_reg[0], VL_reg[1], VL_reg[2], VL_reg[3], gpic_core_num);
                    exit(-1);
                }
                VC_reg = VL_reg[1] * VL_reg[2] * VL_reg[3];
                VM_reg[req.dim].clear();
                VM_reg[req.dim] = std::vector<bool>(req.value);
                fill(VM_reg[req.dim].begin(), VM_reg[req.dim].end(), true);
                if (req.dim == 0) {
                    if (req.value <= 256) {
                        V_PER_SA = (255 / req.value) + 1;
                        SA_PER_V = 1;
                    } else {
                        SA_PER_V = ((req.value - 1) / 256) + 1;
                        V_PER_SA = 1;
                    }
                }
            } else if (req.opcode.find("count") != string::npos) {
                DC_reg = req.value;
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
                hint("All VM_reg[%ld] set to %d\n", req.dim, val);
            } else if (req.opcode.find("only") != string::npos) {
                for (int element = 0; element < VL_reg[req.dim]; element++) {
                    VM_reg[req.dim][element] = !val;
                }
                VM_reg[req.dim][req.value] = val;
                hint("All VM_reg[%ld] set to %d, VM_reg[%ld][%ld] set to %d\n", req.dim, !val, req.dim, req.value, val);
            } else if (req.opcode.find("active") != string::npos) {
                VM_reg[req.dim][req.value] = val;
                hint("VM_reg[%ld][%ld] set to %d\n", req.dim, req.value, val);
            } else {
                assert(false);
            }
        } else {
            assert(false);
        }
        hint("DC_reg: %ld, LS_reg: [%ld, %ld, %ld, %ld], SS_reg: [%ld, %ld, %ld, %ld], VL_reg: [%ld, %ld, %ld, %ld], VC_reg: %ld, V_PER_SA: %d, SA_PER_V: %d\n", DC_reg, LS_reg[0], LS_reg[1], LS_reg[2], LS_reg[3], SS_reg[0], SS_reg[1], SS_reg[2], SS_reg[3], VL_reg[0], VL_reg[1], VL_reg[2], VL_reg[3], VC_reg, V_PER_SA, SA_PER_V);
        // For each vector
        for (int vid_base = 0; vid_base < VC_reg; vid_base += V_PER_SA) {
            // For each SA of the vector
            for (int sid_offset = 0; sid_offset < SA_PER_V; sid_offset++) {
                int sid = vid_to_sid(vid_base, sid_offset);

                // For each vector of the SA
                bool SA_masked_temp = true;
                int remaining_vectors = (V_PER_SA < (VC_reg - vid_base)) ? (V_PER_SA) : (VC_reg - vid_base);
                for (int vid_offset = 0; vid_offset < remaining_vectors; vid_offset++) {
                    // VID shows which address pair should be used
                    int vid = vid_base + vid_offset;

                    if (vector_masked(vid)) {
                        // This vector is masked
                        continue;
                    }

                    SA_masked_temp = false;
                    break;
                }

                SA_masked[sid] = SA_masked_temp;
            }
        }
    } else {

        if (((((VC_reg * SA_PER_V) + 1) / V_PER_SA) - 1) > gpic_core_num) {
            printf("Error: ((((VC_reg(%ld) * SA_PER_V(%d)) + 1) / V_PER_SA(%d)) - 1) (%ld) > gpic_core_num(%d)\n", VC_reg, SA_PER_V, V_PER_SA, ((((VC_reg * SA_PER_V) + 1) / V_PER_SA) - 1), gpic_core_num);
            exit(-1);
        }

        if (check_full_queue(req) == false) {
            hint("SA Queue is full, returning False for %s\n", req.c_str());
            return false;
        }

        if ((req.opcode.find("loadr") != string::npos) || (req.opcode.find("storer") != string::npos) || (req.opcode.find("dict") != string::npos)) {
            random_dict_access_decoder(req);
        } else {
            instrinsic_decoder(req);
        }
    }
    return true;
}

bool Cache::send(Request req) {
    if (req.type == Request::Type::GPIC) {
        hint("level %s received %s\n", level_string.c_str(), req.c_str());

        if (gpic_incoming_req_queue.size() >= MAX_GPIC_QUEUE_SIZE)
            return false;

        gpic_incoming_req_queue.push_back(make_pair(cachesys->clk + latency_each[int(level)], req));
        hint("%s set for start in %d clock cycles\n", req.c_str(), latency_each[int(level)]);

        return true;
    }

    hint("level %s received %s, index %d, tag %ld\n",
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
    std::vector<std::shared_ptr<Line>> &lines = get_lines(req.addr);
    std::shared_ptr<Line> line_ptr;

    if (is_hit(lines, req.addr, &line_ptr)) {

        bool dirty = line_ptr->dirty || (req.type == Request::Type::WRITE);
        long invalidate_time = 0;

        if (req.unitid == (Request::UnitID)(level)) {
            // If it is comming from the same level of the cache, it is produced by a gpic intrinsic

            if (higher_cache.size() != 0) {
                // Make sure it is not L1

                for (auto hc : higher_cache) {
                    std::pair<long, bool> result;
                    if (hc->invalidate(req.addr, result) == false)
                        return false;
                    invalidate_time = max(invalidate_time, result.first + (result.second ? latency_each[int(level)] : 0));
                    dirty = dirty || result.second;

                    if (result.second) {
                        hint("invalidated (%s) from %s due to gpic access\n", req.c_str(), level_string.c_str());
                    }
                }
            }
        }

        hint("%s hitted @level %d\n", req.c_str(), (level));

#ifdef DEBUG
        hint("1- %s: Pushing back addr 0x%lx, idx 0x%x, and tag 0x%lx\n", level_string.c_str(), req.addr, get_index(req.addr), get_tag(req.addr));
        std::map<int, std::vector<std::shared_ptr<Line>>>::iterator cache_line;
        cache_line = cache_lines.find(get_index(req.addr));
        if (cache_line == cache_lines.end()) {
            hint("No line found for the aforementioned idx and tag\n");
        } else {
            std::vector<std::shared_ptr<Line>> liinees;
            liinees = cache_line->second;
            for (std::shared_ptr<Cache::Line> &liinee : liinees) {
                hint("Line addr: 0x%lx tag: 0x%lx\n", liinee->addr, liinee->tag);
            }
        }
#endif

        std::shared_ptr<Cache::Line> line = add_line(&lines, req.addr, false, dirty);

#ifdef DEBUG
        hint("1- %s: Pushed back addr 0x%lx, idx 0x%x, and tag 0x%lx\n", level_string.c_str(), req.addr, get_index(req.addr), get_tag(req.addr));
        cache_line = cache_lines.find(get_index(req.addr));
        if (cache_line == cache_lines.end()) {
            hint("No line found for the aforementioned idx and tag\n");
        } else {
            std::vector<std::shared_ptr<Line>> liinees;
            liinees = cache_line->second;
            for (std::shared_ptr<Cache::Line> &liinee : liinees) {
                hint("Line addr: 0x%lx tag: 0x%lx\n", liinee->addr, liinee->tag);
            }
        }
#endif

        remove_line(&lines, line);

#ifdef DEBUG
        hint("1- %s: Erased addr 0x%lx, idx 0x%x, and tag 0x%lx\n", level_string.c_str(), req.addr, get_index(req.addr), get_tag(req.addr));
        cache_line = cache_lines.find(get_index(req.addr));
        if (cache_line == cache_lines.end()) {
            hint("No line found for the aforementioned idx and tag\n");
        } else {
            std::vector<std::shared_ptr<Line>> liinees;
            liinees = cache_line->second;
            for (std::shared_ptr<Cache::Line> &liinee : liinees) {
                hint("Line addr: 0x%lx tag: 0x%lx\n", liinee->addr, liinee->tag);
            }
        }
#endif

        cachesys->hit_list.push_back(make_pair(cachesys->clk + latency_each[int(level)] + invalidate_time, req));

        hint("hit, update timestamp %ld\n", cachesys->clk);
        hint("hit finish time %ld\n", cachesys->clk + latency_each[int(level)]);

        // Reading/writing block_size bytes from cache for a hit
        cache_access_energy += access_energy;

        return true;
    } else {
        hint("%s missed @level %d\n", req.c_str(), (level));
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
        std::shared_ptr<Line> mshr_entry_line = hit_mshr(req.addr);
        if (mshr_entry_line != nullptr) {
            hint("%s hitted mshr\n", req.c_str());
            cache_mshr_hit++;
            mshr_entry_line->dirty = (dirty || mshr_entry_line->dirty);
            return true;
        }

        // All requests come to this stage will be READ, so they
        // should be recorded in MSHR entries.
        if (mshr_entries.size() == mshr_entry_num) {
            // When no MSHR entries available, the miss request
            // is stalling.
            cache_mshr_unavailable++;
            hint("no mshr entry available\n");
            return false;
        }

        // Check whether there is a line available
        if (all_sets_locked(lines)) {
            cache_set_unavailable++;
            return false;
        }

        std::shared_ptr<Cache::Line> newline = allocate_line(lines, req.addr);
        if (newline == nullptr) {
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

bool Cache::should_send(Request req) {

    // If it's a hit, return true
    if (cache_lines.find(get_index(req.addr)) != cache_lines.end()) {
        std::shared_ptr<Cache::Line> tmp;
        if (is_hit(cache_lines[get_index(req.addr)], req.addr, &tmp)) {
            return true;
        }
    }

    // Look it up in MSHR entries
    std::shared_ptr<Line> mshr_entry_line = hit_mshr(req.addr);
    if (mshr_entry_line != nullptr) {
        return true;
    }

    if (mshr_entries.size() != mshr_entry_num) {
        return true;
    }

    return false;
}

void Cache::evictline(long addr, bool dirty) {

    std::map<int, std::vector<std::shared_ptr<Line>>>::iterator it = cache_lines.find(get_index(addr));
    assert(it != cache_lines.end()); // check inclusive cache
    auto &lines = it->second;

    std::shared_ptr<Line> line_shared_ptr = nullptr;
    for (std::shared_ptr<Line> &line : lines) {
        if (line->tag == get_tag(addr)) {
            line_shared_ptr = line;
            break;
        }
    }

    if (line_shared_ptr == nullptr) {
        printf("%s: addr 0x%lx, idx 0x%x, and tag 0x%lx not found!\n", level_string.c_str(), addr, get_index(addr), get_tag(addr));
        std::map<int, std::vector<std::shared_ptr<Line>>>::iterator cache_line;
        cache_line = cache_lines.find(get_index(addr));
        if (cache_line == cache_lines.end()) {
            hint("No line found for the aforementioned idx and tag\n");
        } else {
            std::vector<std::shared_ptr<Line>> liinees;
            liinees = cache_line->second;
            for (std::shared_ptr<Cache::Line> &liinee : liinees) {
                hint("Line addr: 0x%lx tag: 0x%lx\n", liinee->addr, liinee->tag);
            }
        }
        exit(-2);
    } else {
        hint("line for addr 0x%lx evicted from %s\n", addr, level_string.c_str());
    }

#ifdef DEBUG
    hint("2- %s: Pushing back addr 0x%lx, idx 0x%x, and tag 0x%lx. Before:\n", level_string.c_str(), addr, get_index(addr), get_tag(addr));
    std::map<int, std::vector<std::shared_ptr<Line>>>::iterator cache_line;
    cache_line = cache_lines.find(get_index(addr));
    if (cache_line == cache_lines.end()) {
        hint("No line found for the aforementioned idx and tag\n");
    } else {
        std::vector<std::shared_ptr<Line>> liinees;
        liinees = cache_line->second;
        for (std::shared_ptr<Cache::Line> &liinee : liinees) {
            hint("Line addr: 0x%lx tag: 0x%lx\n", liinee->addr, liinee->tag);
        }
    }
#endif

    // Update LRU queue. The dirty bit will be set if the dirty
    // bit inherited from higher level(s) is set.
    add_line(&lines, addr, false, dirty || line_shared_ptr->dirty);

#ifdef DEBUG
    hint("2- %s: Pushed back addr 0x%lx, idx 0x%x, and tag 0x%lx\n", level_string.c_str(), addr, get_index(addr), get_tag(addr));
    cache_line = cache_lines.find(get_index(addr));
    if (cache_line == cache_lines.end()) {
        hint("No line found for the aforementioned idx and tag\n");
    } else {
        std::vector<std::shared_ptr<Line>> liinees;
        liinees = cache_line->second;
        for (std::shared_ptr<Cache::Line> &liinee : liinees) {
            hint("Line addr: 0x%lx tag: 0x%lx\n", liinee->addr, liinee->tag);
        }
    }
#endif

    remove_line(&lines, line_shared_ptr);

#ifdef DEBUG
    hint("2- %s: Erased addr 0x%lx, idx 0x%x, and tag 0x%lx\n", level_string.c_str(), line_shared_ptr->addr, get_index(line_shared_ptr->addr), get_tag(line_shared_ptr->addr));
    if (cache_line == cache_lines.end()) {
        hint("No line found for the aforementioned idx and tag\n");
    } else {
        std::vector<std::shared_ptr<Line>> liinees;
        liinees = cache_line->second;
        for (std::shared_ptr<Cache::Line> &liinee : liinees) {
            hint("Line addr: 0x%lx tag: 0x%lx\n", liinee->addr, liinee->tag);
        }
    }
#endif
}

bool Cache::invalidate(long addr, std::pair<long, bool> &result) {
    long delay = latency_each[int(level)];
    bool dirty = false;

    std::vector<std::shared_ptr<Line>> &lines = get_lines(addr);
    if (lines.size() == 0) {
        // The line of this address doesn't exist.
        result = make_pair(0, false);
        return true;
    }

    std::shared_ptr<Line> line_shared_ptr = nullptr;
    for (std::shared_ptr<Line> &line : lines) {
        if (line->tag == get_tag(addr)) {
            line_shared_ptr = line;
            break;
        }
    }

    // If the line is in this level cache, then erase it from the buffer.
    if (line_shared_ptr != nullptr) {

        if (line_shared_ptr->lock) {
            return false;
        }
        hint("invalidate 0x%lx @ level %d\n", addr, int(level));
        dirty = line_shared_ptr->dirty;
        remove_line(&lines, line_shared_ptr);

#ifdef DEBUG
        hint("3- %s: Erased addr 0x%lx, idx 0x%x, and tag 0x%lx\n", level_string.c_str(), line_shared_ptr->addr, get_index(line_shared_ptr->addr), line_shared_ptr->tag);
        std::map<int, std::vector<std::shared_ptr<Line>>>::iterator cache_line;
        cache_line = cache_lines.find(get_index(addr));
        if (cache_line == cache_lines.end()) {
            hint("No line found for the aforementioned idx and tag\n");
        } else {
            std::vector<std::shared_ptr<Line>> liinees;
            liinees = cache_line->second;
            for (std::shared_ptr<Cache::Line> &liinee : liinees) {
                hint("Line addr: 0x%lx tag: 0x%lx\n", liinee->addr, liinee->tag);
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

bool Cache::evict(std::vector<std::shared_ptr<Cache::Line>> *lines, std::shared_ptr<Cache::Line> victim) {
    hint("level %d miss evict victim 0x%lx\n", int(level), victim->addr);
    // Before anything, check if this address exists in lower cache
    if (!is_last_level) {
        if (lower_cache->exists_addr(victim->addr) == false) {
            hint("line is not received by the lower cache yet, returning false by evict\n");
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
            if (hc->invalidate(addr, result) == false) {
                return false;
            }
            invalidate_time = max(invalidate_time,
                                  result.first + (result.second ? latency_each[int(level)] : 0));
            dirty = dirty || result.second || victim->dirty;
        }
    }

    hint("invalidate delay: %ld, dirty: %s\n", invalidate_time, dirty ? "true" : "false");

    if (!is_last_level) {
        // not LLC eviction
        assert(lower_cache != nullptr);
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

            hint("inject one write request to memory system "
                 "addr 0x%lx, invalidate time %ld, issue time %ld\n",
                 write_req.addr, invalidate_time,
                 cachesys->clk + invalidate_time + latency_each[int(level)]);
        }
    }

    std::vector<std::shared_ptr<Cache::Line>>::iterator it = lines->begin();
    while (it != lines->end()) {
        if ((*it)->addr == victim->addr) {
            it = lines->erase(it);
        } else {
            ++it;
        }
    }

#ifdef DEBUG
    hint("4- %s: Erased addr 0x%lx, idx 0x%x, and tag 0x%lx\n", level_string.c_str(), victim->addr, get_index(victim->addr), victim->tag);
    std::map<int, std::vector<std::shared_ptr<Line>>>::iterator cache_line;
    cache_line = cache_lines.find(get_index(victim->addr));
    if (cache_line == cache_lines.end()) {
        hint("No line found for the aforementioned idx and tag\n");
    } else {
        std::vector<std::shared_ptr<Line>> liinees;
        liinees = cache_line->second;
        for (std::shared_ptr<Cache::Line> &liinee : liinees) {
            hint("Line addr: 0x%lx tag: 0x%lx\n", liinee->addr, liinee->tag);
        }
    }
#endif

    return true;
}

std::shared_ptr<Cache::Line> Cache::add_line(std::vector<std::shared_ptr<Cache::Line>> *lines, long addr, bool locked, bool dirty) {
    // Allocate newline, with lock bit on and dirty bit off
    Line *line_ptr = new Line(addr, get_tag(addr), locked, dirty);
    std::shared_ptr<Cache::Line> line_shared_ptr = std::shared_ptr<Cache::Line>(line_ptr);
    lines->push_back(line_shared_ptr);
    return line_shared_ptr;
}

bool Cache::remove_line(std::vector<std::shared_ptr<Cache::Line>> *lines, std::shared_ptr<Cache::Line> line) {
    hint("%s: removing line addr 0x%lx, idx 0x%x, and tag 0x%lx\n", level_string.c_str(), line->addr, get_index(line->addr), line->tag);
    std::vector<std::shared_ptr<Cache::Line>>::iterator it = lines->begin();
    bool removed = false;
    while (it != lines->end()) {
        if ((*it)->tag == line->tag) {
            it = lines->erase(it);
            removed = true;
            break;
        } else {
            ++it;
        }
    }
    return removed;
}

std::shared_ptr<Cache::Line> Cache::allocate_line(std::vector<std::shared_ptr<Cache::Line>> &lines, long addr) {
    // See if an eviction is needed
    if (need_eviction(lines, addr)) {
        // Get victim.
        // The first one might still be locked due to reorder in MC
        std::shared_ptr<Cache::Line> victim = nullptr;
        for (std::shared_ptr<Cache::Line> &line : lines) {
            bool check = !line->lock;
            if (!is_first_level) {
                for (auto hc : higher_cache) {
                    if (check == false) {
                        break;
                    }
                    check = check && hc->check_unlock(line->addr);
                }
            }
            if (check == true) {
                victim = line;
                break;
            }
        }
        if (victim == nullptr) {
            return victim; // doesn't exist a line that's already unlocked in each level
        }
        if (evict(&lines, victim) == false)
            return nullptr;
    }

    // Allocate newline, with lock bit on and dirty bit off
    std::shared_ptr<Cache::Line> new_line = add_line(&lines, addr);

#ifdef DEBUG
    hint("3- %s: Pushed back addr 0x%lx, idx 0x%x, and tag 0x%lx\n", level_string.c_str(), addr, get_index(addr), get_tag(addr));
    std::map<int, std::vector<std::shared_ptr<Line>>>::iterator cache_line;
    cache_line = cache_lines.find(get_index(addr));
    if (cache_line == cache_lines.end()) {
        hint("No line found for the aforementioned idx and tag\n");
    } else {
        std::vector<std::shared_ptr<Line>> liinees;
        liinees = cache_line->second;
        for (std::shared_ptr<Cache::Line> &liinee : liinees) {
            hint("Line addr: 0x%lx tag: 0x%lx\n", liinee->addr, liinee->tag);
        }
    }
#endif

    hint("%s addr 0x%lx pushed to lines!\n", level_string.c_str(), addr);

    // Writing block_size bytes to cache as a result of a miss
    cache_access_energy += access_energy;

    return new_line;
}

bool Cache::is_hit(std::vector<std::shared_ptr<Cache::Line>> &lines, long addr, std::shared_ptr<Cache::Line> *pos_ptr) {
    for (std::shared_ptr<Line> &line : lines) {
        if (line->tag == get_tag(addr)) {
            *pos_ptr = line;
            return !line->lock;
        }
    }
    return false;
}

void Cache::concatlower(Cache *lower) {
    lower_cache = lower;
    assert(lower != nullptr);
    lower->higher_cache.push_back(this);
};

bool Cache::need_eviction(std::vector<std::shared_ptr<Cache::Line>> &lines, long addr) {
    for (std::shared_ptr<Cache::Line> &line : lines) {
        // Due to MSHR, the program can't reach here. Just for checking
        assert(line->tag != get_tag(addr));
    }
    if (lines.size() < assoc) {
        return false;
    } else {
        return true;
    }
}

void Cache::callback(Request &req) {
    hint("level %d\n", int(level));
    hint("%s received in %s\n", req.c_str(), level_string.c_str());

    // Remove related MSHR entries
    int mshr_idx = 0;
    for (; mshr_idx < mshr_entries.size(); mshr_idx++) {
        if (align(mshr_entries[mshr_idx].first) == align(req.addr)) {
            break;
        }
    }

    if (mshr_idx != mshr_entries.size()) {
        assert(mshr_entries[mshr_idx].second != NULL);
        mshr_entries[mshr_idx].second->lock = false;
        hint("pair(0x%lx, line(0x%lx, %d, %d, 0x%lx)) removed from mshr entries at level %s\n", mshr_entries[mshr_idx].first, mshr_entries[mshr_idx].second->addr, mshr_entries[mshr_idx].second->dirty, mshr_entries[mshr_idx].second->lock, mshr_entries[mshr_idx].second->tag, level_string.c_str());
        mshr_entries.erase(mshr_entries.begin() + mshr_idx);
    } else {
        hint("NO MSHR entry removed at %s\n", level_string.c_str());
    }

    // Remove corresponding random load/store addresses
    auto random_it = gpic_random_dict_to_mem_ops.begin();
    while (random_it != gpic_random_dict_to_mem_ops.end()) {
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
            hint("unlocking...\n");
            locked = false;
            random_it = gpic_random_dict_to_mem_ops.erase(random_it);
            gpic_incoming_req_queue.erase(gpic_incoming_req_queue.begin());
        } else {
            ++random_it;
        }
    }

    // Remove corresponding GPIC instructions
    int sid_start = rand() % gpic_core_num;
    for (int sid_offset = 0; sid_offset < gpic_core_num; sid_offset++) {
        int sid = (sid_start + sid_offset) % gpic_core_num;

        bool hit = false;

        // Check if the SA has sent the memory operations
        if ((last_gpic_instruction_computed[sid] == true) && (last_gpic_instruction_sent[sid] == true)) {
            Request gpic_req = gpic_compute_queue[sid][0].second;

            // Check all start-end address pairs
            for (int gpic_idx = 0; gpic_idx < gpic_req.addr_starts.size(); gpic_idx++) {

                // If the address has overlap with the start-end pair
                if ((align(req.addr) >= align(gpic_req.addr_starts[gpic_idx])) && (align(req.addr) <= align(gpic_req.addr_ends[gpic_idx]))) {

                    // Check if this memory access has been ocurred because of this GPIC instruction
                    auto iter = gpic_op_to_mem_ops[sid][gpic_req].begin();
                    while (iter != gpic_op_to_mem_ops[sid][gpic_req].end()) {
                        if (iter->second == false) {
                            ++iter;
                            continue;
                        }
                        if (align(req.addr) == align(iter->first.addr)) {
                            hint("1- %s: %s calls back for %s, %lu instructions remained\n", level_string.c_str(), req.c_str(), gpic_req.c_str(), gpic_op_to_mem_ops[sid][gpic_req].size() - 1);
                            hit = true;
                            // Remove this instruction from gpic list
                            iter = gpic_op_to_mem_ops[sid][gpic_req].erase(iter);
                        } else {
                            ++iter;
                        }
                    }
                }
            }
            if (gpic_op_to_mem_ops[sid][gpic_req].size() == 0) {
                op_trace << cachesys->clk << " " << sid << " F " << gpic_req.opcode << endl;
                hint("18- %s: calling back %s\n", level_string.c_str(), gpic_req.c_str());
                callbacker(gpic_req);
                gpic_op_to_mem_ops[sid].erase(gpic_req);
                gpic_compute_queue[sid].erase(gpic_compute_queue[sid].begin());
                hint("Compute queue [%d] removed size: %d\n", sid, (int)gpic_compute_queue[sid].size());

                last_gpic_instruction_compute_clk[sid] = -1;
                last_gpic_instruction_computed[sid] = false;
                last_gpic_instruction_sent[sid] = false;
            } else if (hit == true) {
                // send the first non-sent addresses for other queues until mshr is full again
                // bool mshr_full = false;
                // for (int sid_idx = sid; sid_idx < gpic_core_num; sid_idx++) {

                //     // Check if the SA has sent the memory operations
                //     if ((last_gpic_instruction_computed[sid_idx] == true) && (last_gpic_instruction_sent[sid_idx] == true)) {
                //         Request gpic_idx_req = gpic_compute_queue[sid_idx][0].second;

                //         // If the same instruction is at the head of the other SA's queue
                //         if (gpic_idx_req.reqid == gpic_req.reqid) {

                for (int mem_idx = 0; mem_idx < gpic_op_to_mem_ops[sid][gpic_req].size(); mem_idx++) {
                    if (gpic_op_to_mem_ops[sid][gpic_req][mem_idx].second == true)
                        continue;

                    hint("15- %s sending %s to %s\n", level_string.c_str(), gpic_op_to_mem_ops[sid][gpic_req][mem_idx].first.c_str(), level_string.c_str());
                    bool should = should_send(gpic_op_to_mem_ops[sid][gpic_req][mem_idx].first);
                    bool sent = false;
                    if (should == true) {
                        sent = send(gpic_op_to_mem_ops[sid][gpic_req][mem_idx].first);
                    }
                    if ((should == false) || (sent == false)) {
                        hint("1- should (%d) or sent (%d) is false!\n", should, sent);
                        self_retry_list.push_back(gpic_op_to_mem_ops[sid][gpic_req][mem_idx].first);
                        gpic_op_to_mem_ops[sid][gpic_req][mem_idx].second = true;
                        break;
                    }
                    gpic_op_to_mem_ops[sid][gpic_req][mem_idx].second = true;
                }
                //         }
                //     }

                //     if (mshr_full == true)
                //         break;
                // }
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
    assert(gpic_vop_to_num_sop.count(req) == 1);
    assert(gpic_vop_to_num_sop[req] > 0);
    gpic_vop_to_num_sop[req] -= 1;
    if (gpic_vop_to_num_sop[req] == 0) {
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
        bool should = should_send(self_retry_list[0]);
        bool sent = false;
        if (should == true) {
            sent = send(self_retry_list[0]);
        }
        if ((should == false) || (sent == false)) {
            hint("2- should (%d) or sent (%d) is false!\n", should, sent);
            break;
        }
        self_retry_list.erase(self_retry_list.begin());
    }

    if (locked == false) {
        while ((gpic_incoming_req_queue.size() > 0) && (cachesys->clk >= gpic_incoming_req_queue[0].first)) {
            if (memory_controller(gpic_incoming_req_queue[0].second) == false)
                break;
            if (locked == false)
                gpic_incoming_req_queue.erase(gpic_incoming_req_queue.begin());
            else
                break;
        }
    }

    // Instruction at the head of the GPIC queue is computed
    // If it's a store or load it is unpacked
    // Otherwise, it is called back
    int sid_start = rand() % gpic_core_num;
    for (int sid_offset = 0; sid_offset < gpic_core_num; sid_offset++) {
        int sid = (sid_start + sid_offset) % gpic_core_num;

        // Check if there is any instruction ready for completion
        if ((last_gpic_instruction_computed[sid] == true) && (last_gpic_instruction_sent[sid] == false)) {

            // There must be an instruction going on
            assert(gpic_compute_queue[sid].size() != 0);

            // Check if the instruction is done
            if (cachesys->clk - last_gpic_instruction_compute_clk[sid] >= gpic_compute_queue[sid].at(0).first) {
                Request req = gpic_compute_queue[sid].at(0).second;

                if ((req.opcode.find("load") != string::npos) || (req.opcode.find("store") != string::npos)) {
                    // If it's a load or store, make new queries and send to this cache level's queue

                    hint("Unpacking loads/stores for: %s\n", req.c_str());

                    assert(gpic_op_to_mem_ops[sid].count(req) == 0);
                    gpic_op_to_mem_ops[sid][req] = std::vector<std::pair<Request, bool>>();
                    gpic_op_to_mem_ops[sid][req].clear();
                    for (int idx = 0; idx < req.addr_starts.size(); idx++) {
                        if (req.addr_starts[idx] == 0) {
                            hint("13- %s ignored one zero-addressed memory access for %s\n", level_string.c_str(), req.c_str());
                            continue;
                        }
                        long addr = req.addr_starts[idx];
                        for (int i = req.min_vid; i < req.max_vid; i++) {
                            if ((addr < req.addr_starts[idx]) || (addr > req.addr_ends[idx])) {
                                printf("Error: (addr(0x%lx) < req.addr_starts[%d](0x%lx)) || (addr(0x%lx) > req.addr_ends[%d](0x%lx))", addr, i, req.addr_starts[idx], addr, i, req.addr_ends[idx]);
                                exit(-1);
                            }
                            if (req.vector_mask.size() <= i) {
                                printf("mask size (%ld) <= i (%d), for req: %s\n", req.vector_mask.size(), i, req.c_str());
                                exit(-2);
                            }
                            if (req.vector_mask[i]) {
                                if (addr_exists(gpic_op_to_mem_ops[sid][req], align(addr))) {
                                    hint("14- %s NOT sending 0x%lx to %s\n", level_string.c_str(), align(addr), level_string.c_str());
                                } else {
                                    // make the request
                                    Request::Type req_type = (req.opcode.find("load") != string::npos) ? Request::Type::READ : Request::Type::WRITE;
                                    int req_coreid = req.coreid;
                                    Request::UnitID req_unitid = (Request::UnitID)(level);
                                    Request mem_req(align(addr), req_type, true, processor_callback, req_coreid, req_unitid);
                                    mem_req.reqid = last_id;
                                    last_id++;
                                    gpic_op_to_mem_ops[sid][req].push_back(std::pair<Request, bool>(mem_req, false));
                                    hint("unpacked: %s\n", mem_req.c_str());
                                }
                            }
                            addr += (req.stride * req.data_type / 8);
                        }
                    }

                    last_gpic_instruction_sent[sid] = true;

                    if (gpic_op_to_mem_ops[sid][req].size() == 0) {
                        op_trace << cachesys->clk << " " << sid << " F " << req.opcode << endl;
                        hint("17- %s: calling back %s\n", level_string.c_str(), req.c_str());
                        callbacker(req);
                        gpic_op_to_mem_ops[sid].erase(req);
                        gpic_compute_queue[sid].erase(gpic_compute_queue[sid].begin());
                        last_gpic_instruction_compute_clk[sid] = -1;
                        last_gpic_instruction_computed[sid] = false;
                        last_gpic_instruction_sent[sid] = false;
                    } else {
                        // send the until mshr is full
                        for (int mem_idx = 0; mem_idx < gpic_op_to_mem_ops[sid][req].size(); mem_idx++) {
                            hint("15- %s sending %s to %s\n", level_string.c_str(), gpic_op_to_mem_ops[sid][req][mem_idx].first.c_str(), level_string.c_str());

                            bool should = should_send(gpic_op_to_mem_ops[sid][req][mem_idx].first);
                            bool sent = false;
                            if (should == true) {
                                sent = send(gpic_op_to_mem_ops[sid][req][mem_idx].first);
                            }
                            if ((should == false) || (sent == false)) {
                                hint("3- should (%d) or sent (%d) is false!\n", should, sent);
                                if (mem_idx == 0) {
                                    self_retry_list.push_back(gpic_op_to_mem_ops[sid][req][mem_idx].first);
                                    gpic_op_to_mem_ops[sid][req][mem_idx].second = true;
                                }
                                break;
                            }
                            gpic_op_to_mem_ops[sid][req][mem_idx].second = true;
                        }
                    }

                } else {
                    // Otherwise, we are ready to send the call back
                    op_trace << cachesys->clk << " " << sid << " F " << req.opcode << endl;

                    hint("16- %s instruction %s completed\n", level_string.c_str(), req.c_str());
                    callbacker(req);
                    gpic_compute_queue[sid].erase(gpic_compute_queue[sid].begin());
                    hint("Compute queue [%d] removed size: %d\n", sid, (int)gpic_compute_queue[sid].size());
                    last_gpic_instruction_compute_clk[sid] = -1;
                    last_gpic_instruction_computed[sid] = false;
                    last_gpic_instruction_sent[sid] = false;
                }
            }
        }
    }

    // Instruction at the head of the GPIC queue gets ready for compute
    sid_start = rand() % gpic_core_num;
    for (int sid_offset = 0; sid_offset < gpic_core_num; sid_offset++) {
        int sid = (sid_start + sid_offset) % gpic_core_num;
        // Check if there is any instructions ready to be computed
        if (last_gpic_instruction_computed[sid] == false) {

            // The last instruction must not be sent
            assert(last_gpic_instruction_sent[sid] == false);

            if (gpic_compute_queue[sid].size() != 0) {
                // A new instruction must be computed
                if (gpic_compute_queue[sid].at(0).second.opcode.find("move") != string::npos) {

                    // we should wait for the dst SA as well
                    int sid_src = gpic_compute_queue[sid].at(0).second.sid;
                    int sid_dst = gpic_compute_queue[sid].at(0).second.sid_dst;
                    assert(sid_dst != -1);

                    if (sid_src == sid_dst) {
                        op_trace << cachesys->clk << " " << sid << " S " << gpic_compute_queue[sid].at(0).second.opcode << endl;
                        hint("Computing %s at %ld, %zu instructions in compute queue\n", gpic_compute_queue[sid].at(0).second.c_str(), cachesys->clk, gpic_compute_queue[sid].size());
                        last_gpic_instruction_compute_clk[sid] = cachesys->clk;
                        last_gpic_instruction_computed[sid] = true;
                    } else if (sid != sid_dst) {
                        // only src SA checks for the dst, not vice versa

                        // Check if there is any instructions ready to be computed in the dst queue
                        if (last_gpic_instruction_computed[sid_dst] == false) {

                            // The last instruction must not be sent
                            assert(last_gpic_instruction_sent[sid_dst] == false);

                            if (gpic_compute_queue[sid_dst].size() != 0) {

                                // Check if these are the same instructions
                                if (gpic_compute_queue[sid].at(0).second == gpic_compute_queue[sid_dst].at(0).second) {

                                    // Compute both
                                    op_trace << cachesys->clk << " " << sid << " S " << gpic_compute_queue[sid].at(0).second.opcode << endl;
                                    op_trace << cachesys->clk << " " << sid_dst << " S " << gpic_compute_queue[sid].at(0).second.opcode << endl;
                                    hint("Computing %s at %ld, %zu instructions in compute queue\n", gpic_compute_queue[sid].at(0).second.c_str(), cachesys->clk, gpic_compute_queue[sid].size());
                                    hint("Computing %s at %ld, %zu instructions in compute queue\n", gpic_compute_queue[sid_dst].at(0).second.c_str(), cachesys->clk, gpic_compute_queue[sid_dst].size());
                                    last_gpic_instruction_compute_clk[sid] = cachesys->clk;
                                    last_gpic_instruction_compute_clk[sid_dst] = cachesys->clk;
                                    last_gpic_instruction_computed[sid] = true;
                                    last_gpic_instruction_computed[sid_dst] = true;
                                }
                            }
                        }

                        if (last_gpic_instruction_computed[sid] == false) {
                            hint("%s is waiting for dst SA\n", gpic_compute_queue[sid].at(0).second.c_str());
                        }
                    }

                } else {
                    op_trace << cachesys->clk << " " << sid << " S " << gpic_compute_queue[sid].at(0).second.opcode << endl;
                    hint("Computing %s at %ld, %zu instructions in compute queue\n", gpic_compute_queue[sid].at(0).second.c_str(), cachesys->clk, gpic_compute_queue[sid].size());
                    last_gpic_instruction_compute_clk[sid] = cachesys->clk;
                    last_gpic_instruction_computed[sid] = true;
                }
            }
        }
    }

    sid_start = rand() % gpic_core_num;
    for (int sid_offset = 0; sid_offset < gpic_core_num; sid_offset++) {
        int sid = (sid_start + sid_offset) % gpic_core_num;
        if ((last_gpic_instruction_computed[sid] == false) && (last_gpic_instruction_sent[sid] == false)) {

            if (gpic_compute_queue[sid].size() == 0) {
                // There is no instruction ready for execute
                if (level == Level::L2) {
                    if (sid == 0) {
                        hint("%s GPIC %d HOST_DEVICE...\n", level_string.c_str(), sid);
                    }
                }
                GPIC_host_device_total_cycles++;
                GPIC_host_device_cycles[sid]++;
            } else {
                // Instruction at top must be move and stalled by dst SA
                // hint("%s GPIC %d MOVE...\n", level_string.c_str(), sid);
                assert(gpic_compute_queue[sid].at(0).second.opcode.find("move") != string::npos);
                GPIC_move_stall_total_cycles++;
                GPIC_move_stall_cycles[sid]++;
            }
        }

        if ((last_gpic_instruction_computed[sid] == true) && (last_gpic_instruction_sent[sid] == false)) {
            // An instruction is being computed
            // hint("%s GPIC %d COMPUTE...\n", level_string.c_str(), sid);
            GPIC_compute_total_cycles++;
            GPIC_compute_cycles[sid]++;
        }

        if ((last_gpic_instruction_computed[sid] == true) && (last_gpic_instruction_sent[sid] == true)) {
            // An instruction is computed and sent, waiting for memory instructions to be called back
            // hint("%s GPIC %d MEMORY...\n", level_string.c_str(), sid);
            GPIC_memory_total_cycles++;
            GPIC_memory_cycles[sid]++;
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

    if (gpic_incoming_req_queue.size() != 0)
        return false;

    if (gpic_random_dict_to_mem_ops.size() != 0)
        return false;

    for (int sid = 0; sid < gpic_core_num; sid++) {
        if (gpic_op_to_mem_ops[sid].size() != 0)
            return false;

        if (gpic_compute_queue[sid].size() != 0)
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
    GPIC_host_device_total_cycles = 0;
    GPIC_move_stall_total_cycles = 0;
    GPIC_compute_total_cycles = 0;
    GPIC_memory_total_cycles = 0;
    GPIC_compute_total_energy = 0;
    GPIC_compute_comp_total_energy = 0;
    GPIC_compute_rdwr_total_energy = 0;
    assert(gpic_random_dict_to_mem_ops.size() == 0);

    for (int i = 0; i < gpic_incoming_req_queue.size(); i++) {
        printf("ERROR: %s remained in gpic incoming instruction queue %s\n", gpic_incoming_req_queue.at(0).second.c_str(), level_string.c_str());
    }
    assert(gpic_incoming_req_queue.size() == 0);

    for (int sid = 0; sid < gpic_core_num; sid++) {
        GPIC_host_device_cycles[sid] = 0;
        GPIC_move_stall_cycles[sid] = 0;
        GPIC_compute_cycles[sid] = 0;
        GPIC_memory_cycles[sid] = 0;
        GPIC_compute_energy[sid] = 0;
        GPIC_compute_comp_energy[sid] = 0;
        GPIC_compute_rdwr_energy[sid] = 0;
        last_gpic_instruction_computed[sid] = false;
        last_gpic_instruction_sent[sid] = false;
        assert(gpic_op_to_mem_ops[sid].size() == 0);
        SA_masked[sid] = false;

        for (int i = 0; i < gpic_compute_queue[sid].size(); i++) {
            printf("ERROR: %s remained in gpic compute queue %s\n", gpic_compute_queue[sid].at(0).second.c_str(), level_string.c_str());
        }
        assert(gpic_compute_queue[sid].size() == 0);
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

    SA_PER_V = gpic_core_num;
    DC_reg = 1;
    VL_reg[0] = gpic_core_num * 256;
    VL_reg[1] = VL_reg[2] = VL_reg[3] = 1;
    VC_reg = 1;
    LS_reg[0] = LS_reg[1] = LS_reg[2] = LS_reg[3] = 0;
    SS_reg[0] = SS_reg[1] = SS_reg[2] = SS_reg[3] = 0;
    VM_reg[0] = vector<bool>(gpic_core_num * 256);
    fill(VM_reg[0].begin(), VM_reg[0].end(), true);
    VM_reg[1] = vector<bool>(1);
    VM_reg[1][0] = true;
    VM_reg[2] = vector<bool>(1);
    VM_reg[2][0] = true;
    VM_reg[3] = vector<bool>(1);
    VM_reg[3][0] = true;
}

void CacheSystem::tick() {
    hint("clk %ld\n", clk);

    ++clk;

    // Sends ready waiting request to memory
    auto it = wait_list.begin();
    while (it != wait_list.end() && clk >= it->first) {
        if (!send_memory(it->second)) {

            hint("failed sending %s to memory\n", (it->second).c_str());

            ++it;
        } else {

            hint("complete req: %s\n", (it->second).c_str());

            it = wait_list.erase(it);
        }
    }

    // hit request callback
    it = hit_list.begin();
    while (it != hit_list.end()) {
        if (clk >= it->first) {
            it->second.callback(it->second);

            hint("finish hit: %s\n", (it->second).c_str());

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
        hint("%s\n", it->second.c_str());
        ++it;
    }
    assert(wait_list.size() == 0);

    it = hit_list.begin();
    while (it != hit_list.end()) {
        hint("%s\n", it->second.c_str());
        ++it;
    }
    assert(hit_list.size() == 0);
}

bool CacheSystem::finished() {
    return (wait_list.size() == 0) && (hit_list.size() == 0);
}

} // namespace ramulator

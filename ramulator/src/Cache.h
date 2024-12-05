#ifndef __CACHE_H
#define __CACHE_H

#include "Config.h"
#include "Request.h"
#include "Statistics.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <math.h>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define MAX_MVE_QUEUE_SIZE 32

namespace ramulator {
class CacheSystem;

// These variables are set in Main.cpp
extern std::map<core_type_t, core_config_t> core_configs;

extern int l3_size;
extern int l3_assoc;
extern int l3_blocksz;
extern int mshr_per_bank;
extern float l3_access_energy;
extern int l3_MVE_SA_num;

class Cache {
protected:
    ScalarStat cache_read_miss;
    ScalarStat cache_write_miss;
    ScalarStat cache_total_miss;
    ScalarStat cache_eviction;
    ScalarStat cache_read_access;
    ScalarStat cache_write_access;
    ScalarStat cache_total_access;
    ScalarStat cache_mshr_hit;
    ScalarStat cache_mshr_unavailable;
    ScalarStat cache_set_unavailable;
    ScalarStat cache_access_energy;

    ScalarStat *MVE_compute_energy;
    ScalarStat *MVE_compute_comp_energy;
    ScalarStat *MVE_compute_rdwr_energy;
    ScalarStat MVE_compute_total_energy;
    ScalarStat MVE_compute_comp_total_energy;
    ScalarStat MVE_compute_rdwr_total_energy;

    ScalarStat MVE_host_device_total_cycles;
    ScalarStat MVE_move_stall_total_cycles;
    ScalarStat MVE_compute_total_cycles;
    ScalarStat MVE_memory_total_cycles;
    ScalarStat *MVE_host_device_cycles;
    ScalarStat *MVE_move_stall_cycles;
    ScalarStat *MVE_compute_cycles;
    ScalarStat *MVE_memory_cycles;

public:
    enum class Level {
        CORE,
        L1,
        L2,
        L3,
        MAX
    } level;
    std::string level_string;

    struct Line {
        long addr;
        long tag;
        bool lock; // When the lock is on, the value is not valid yet.
        bool dirty;
        Line(long addr, long tag)
            : addr(addr), tag(tag), lock(true), dirty(false) {
        }
        Line(long addr, long tag, bool lock, bool dirty)
            : addr(addr), tag(tag), lock(lock), dirty(dirty) {
        }
    };

    Cache(int size, int assoc, int block_size, int mshr_entry_num, float access_energy,
          Level level, std::shared_ptr<CacheSystem> cachesys, int MVE_CB_num, int core_id = -1);

    void tick();
    void reset_state();

    // L1, L2, L3 accumulated latencies
    int latency_each[int(Level::MAX)] = {0, 4, 12, 31};

    std::shared_ptr<CacheSystem> cachesys;
    // LLC has multiple higher caches
    std::vector<Cache *> higher_cache;
    Cache *lower_cache;

    bool send(Request req);

    void concatlower(Cache *lower);

    void callback(Request &req);

    function<void(Request &)> processor_callback;

    bool finished();

    int vid_to_sid(int vid, int base);

    bool check_full_queue(Request req);

    bool MVE_controller(Request req);

    void instrinsic_decoder(Request req);

    void intrinsic_computer(Request req);

    int stride_evaluator(long rstride, bool load);

    void random_dict_access_decoder(Request req);

    void callbacker(Request &req);

    bool vector_masked(int vid);

    // Check whether this addr is hit and fill in the pos_ptr with the iterator to the hit line or lines.end()
    bool is_hit(std::vector<std::shared_ptr<Line>> &lines, long addr, std::shared_ptr<Line> *pos_ptr);

    std::shared_ptr<Cache::Line> add_line(std::vector<std::shared_ptr<Line>> *lines, long addr, bool locked = true, bool dirty = false);

    bool remove_line(std::vector<std::shared_ptr<Line>> *lines, std::shared_ptr<Line> line);

    bool should_send(Request req);

    // Align the address to cache line size
    long align(long addr) {
        return (addr & ~(block_size - 1l));
    }

protected:
    int core_id;
    bool is_first_level;
    bool is_last_level;
    size_t size;
    unsigned int assoc;
    unsigned int block_num;
    unsigned int index_mask;
    unsigned int block_size;
    unsigned int index_offset;
    unsigned int tag_offset;
    unsigned int mshr_entry_num;
    float access_energy;
    long last_id = 0;
    // Number of MVE Controller Blocks (CB)
    int MVE_CB_num;
    bool receive_locked = false;
    bool crossbar_locked = false;

    std::vector<std::pair<long, std::shared_ptr<Line>>> mshr_entries;

    std::vector<Request> self_retry_list;

    std::list<std::pair<long, Request>> retry_list;

    std::map<int, std::vector<std::shared_ptr<Line>>> cache_lines;

    // In-cache compute and data access delays
    // Taken from the CSV files in data directory
    std::map<std::string, long> MVE_COMPUTE_DELAY;
    std::map<std::string, long> MVE_ACCESS_DELAY;

    void init_intrinsic_latency();

    // Map of MVE instructions to memory operations
    std::map<Request, std::vector<std::pair<Request, bool>>> MVE_op_to_mem_ops[MAX_MVE_SA_NUM];
    std::map<Request, std::vector<long>> MVE_random_dict_to_mem_ops;
    std::vector<std::pair<long, Request>> MVE_incoming_req_queue;
    std::vector<std::pair<long, Request>> MVE_compute_queue[MAX_MVE_SA_NUM];
    long last_MVE_instruction_compute_clk[MAX_MVE_SA_NUM];
    bool last_MVE_instruction_computed[MAX_MVE_SA_NUM];
    bool last_MVE_instruction_sent[MAX_MVE_SA_NUM];
    std::map<Request, int> MVE_vop_to_num_sop;

    // Vector length for 4 dimensions
    long VL_reg[4];

    // Vector count register = VL_reg[1] * VL_reg[2] * VL_reg[3]
    long VC_reg = 1;

    // Dimension count register
    long DC_reg = 1;

    // Vector mask register for each element of each dimension
    bool *VM_reg[4];

    // Load stride register for each dimension
    long LS_reg[4] = {0, 0, 0, 0};

    // Store stride register for each dimension
    long SS_reg[4] = {0, 0, 0, 0};

#if ISA_TYPE == RVV_ISA
    // If a vector lane is masked
    bool V_masked[MAX_MVE_SA_NUM * LANES_PER_SA];
#else
    // If a control block is masked
    bool CB_masked[MAX_MVE_SA_NUM * LANES_PER_SA];
#endif

    // Number of vectors per control block
    int V_PER_CB = 1;
    // Number of control blocks per vector
    int CB_PER_V = MAX_MVE_SA_NUM;

    int calc_log2(int val) {
        int n = 0;
        while ((val >>= 1))
            n++;
        return n;
    }

    int get_index(long addr) {
        return (addr >> index_offset) & index_mask;
    };

    long get_tag(long addr) {
        return (addr >> tag_offset);
    }

    // Evict the cache line from higher level to this level.
    // Pass the dirty bit and update LRU queue.
    void evictline(long addr, bool dirty);

    // Invalidate the line from this level to higher levels
    // The return value is a pair. The first element is invalidation
    // latency, and the second is wether the value has new version
    // in higher level and this level.
    bool invalidate(long addr, std::pair<long, bool> &result);

    // Evict the victim from current set of lines.
    // First do invalidation, then call evictline(L1 or L2) or send
    // a write request to memory(L3) when dirty bit is on.
    bool evict(std::vector<std::shared_ptr<Cache::Line>> *lines, std::shared_ptr<Cache::Line> victim);

    // First test whether need eviction, if so, do eviction by
    // calling evict function. Then allocate a new line and return
    // the iterator points to it.
    std::shared_ptr<Line> allocate_line(std::vector<std::shared_ptr<Cache::Line>> &lines, long addr);

    // Check whether the set to hold addr has space or eviction is
    // needed.
    bool need_eviction(std::vector<std::shared_ptr<Cache::Line>> &lines, long addr);

    bool exists_addr(long addr) {
        std::map<int, std::vector<std::shared_ptr<Line>>>::iterator it;
        it = cache_lines.find(get_index(addr));
        if (it == cache_lines.end()) {
            return false;
        } else {
            std::vector<std::shared_ptr<Line>> &lines = it->second;
            for (std::shared_ptr<Line> &line : lines) {
                if (line->tag == get_tag(addr)) {
                    return true;
                }
            }
            return false;
        }
    }

    bool all_sets_locked(std::vector<std::shared_ptr<Cache::Line>> &lines) {
        if (lines.size() < assoc) {
            return false;
        }
        for (std::shared_ptr<Cache::Line> &line : lines) {
            if (!line->lock) {
                return false;
            }
        }
        return true;
    }

    bool check_unlock(long addr) {
        std::map<int, std::vector<std::shared_ptr<Line>>>::iterator it;
        it = cache_lines.find(get_index(addr));
        if (it == cache_lines.end()) {
            return true;
        } else {
            std::vector<std::shared_ptr<Line>> &lines = it->second;
            std::shared_ptr<Line> line_ptr = nullptr;
            for (std::shared_ptr<Line> &line : lines) {
                if (line->tag == get_tag(addr)) {
                    line_ptr = line;
                    break;
                }
            }
            if (line_ptr == nullptr) {
                return true;
            } else {
                bool check = !line_ptr->lock;
                if (!is_first_level) {
                    for (auto hc : higher_cache) {
                        if (!check) {
                            return check;
                        }
                        check = check && hc->check_unlock(line_ptr->addr);
                    }
                }
                return check;
            }
        }
    }

    std::shared_ptr<Line> hit_mshr(long addr) {
        std::vector<std::pair<long, std::shared_ptr<Line>>>::iterator mshr_it;
        mshr_it = find_if(mshr_entries.begin(), mshr_entries.end(),
                          [addr, this](std::pair<long, std::shared_ptr<Line>> mshr_entry) {
                              return (align(mshr_entry.first) == align(addr));
                          });
        if (mshr_it == mshr_entries.end())
            return nullptr;
        return mshr_it->second;
    }

    std::vector<std::shared_ptr<Line>> &get_lines(long addr) {
        if (cache_lines.find(get_index(addr)) == cache_lines.end()) {
            cache_lines.insert(make_pair(get_index(addr), std::vector<std::shared_ptr<Line>>()));
        }
        return cache_lines[get_index(addr)];
    }
};

class CacheSystem {
public:
    CacheSystem(const Config &configs, std::function<bool(Request)> send_memory)
        : send_memory(send_memory) {
        if (configs.has_core_caches()) {
            first_level = Cache::Level::L1;
        } else if (configs.has_l3_cache()) {
            first_level = Cache::Level::L3;
        } else {
            last_level = Cache::Level::MAX; // no cache
        }

        if (configs.has_l3_cache()) {
            last_level = Cache::Level::L3;
        } else if (configs.has_core_caches()) {
            last_level = Cache::Level::L2;
        } else {
            last_level = Cache::Level::MAX; // no cache
        }
    }

    // wait_list contains miss requests with their latencies in
    // cache. When this latency is met, the send_memory function
    // will be called to send the request to the memory system.
    std::list<std::pair<long, Request>> wait_list;

    // hit_list contains hit requests with their latencies in cache.
    // callback function will be called when this latency is met and
    // set the instruction status to ready in processor's window.
    std::list<std::pair<long, Request>> hit_list;

    std::function<bool(Request)> send_memory;

    long clk = 0;
    void tick();
    void reset_state();
    bool finished();

    Cache::Level first_level;
    Cache::Level last_level;
};

} // namespace ramulator

#endif /* __CACHE_H */

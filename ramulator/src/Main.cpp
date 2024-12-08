#include "Config.h"
#include "Controller.h"
#include "DRAM.h"
#include "Memory.h"
#include "Processor.h"
#include "Request.h"
#include "SpeedyController.h"
#include "Statistics.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <stdlib.h>
#include <string>
#include <vector>

/* Standards */
#include "ALDRAM.h"
#include "DDR3.h"
#include "DDR4.h"
#include "DSARP.h"
#include "GDDR5.h"
#include "Gem5Wrapper.h"
#include "HBM.h"
#include "LPDDR3.h"
#include "LPDDR4.h"
#include "PCM.h"
#include "SALP.h"
#include "STTMRAM.h"
#include "TLDRAM.h"
#include "WideIO.h"
#include "WideIO2.h"

using namespace std;
using namespace ramulator;

bool ramulator::warmup_complete = false;

std::map<core_type_t, core_config_t> ramulator::core_configs;

int ramulator::l3_size = 1 << 22;
int ramulator::l3_assoc = 1 << 8;
int ramulator::l3_blocksz = 1 << 6;
int ramulator::mshr_per_bank = 64;
float ramulator::l3_access_energy = 117.20;
int ramulator::l3_MVE_SA_num = 32;

void declare_configuration(const Config &configs) {
    ramulator::core_configs[core_type_t::SILVER].l1_cache_config.size = 1 << 16;
    ramulator::core_configs[core_type_t::SILVER].l1_cache_config.assoc = 1 << 2;
    ramulator::core_configs[core_type_t::SILVER].l1_cache_config.blocksz = 1 << 6;
    ramulator::core_configs[core_type_t::SILVER].l1_cache_config.mshr_num = 12;
    ramulator::core_configs[core_type_t::SILVER].l1_cache_config.access_energy = 5.00;

    ramulator::core_configs[core_type_t::GOLD].l1_cache_config.size = 1 << 16;
    ramulator::core_configs[core_type_t::GOLD].l1_cache_config.assoc = 1 << 2;
    ramulator::core_configs[core_type_t::GOLD].l1_cache_config.blocksz = 1 << 6;
    ramulator::core_configs[core_type_t::GOLD].l1_cache_config.mshr_num = 12;
    ramulator::core_configs[core_type_t::GOLD].l1_cache_config.access_energy = 5.00;

    ramulator::core_configs[core_type_t::PRIME].l1_cache_config.size = 1 << 16;
    ramulator::core_configs[core_type_t::PRIME].l1_cache_config.assoc = 1 << 2;
    ramulator::core_configs[core_type_t::PRIME].l1_cache_config.blocksz = 1 << 6;
    ramulator::core_configs[core_type_t::PRIME].l1_cache_config.mshr_num = 12;
    ramulator::core_configs[core_type_t::PRIME].l1_cache_config.access_energy = 5.00;

    ramulator::core_configs[core_type_t::SILVER].l2_cache_config.size = 1 << 16;
    ramulator::core_configs[core_type_t::SILVER].l2_cache_config.assoc = 1 << 2;
    ramulator::core_configs[core_type_t::SILVER].l2_cache_config.blocksz = 1 << 6;
    ramulator::core_configs[core_type_t::SILVER].l2_cache_config.mshr_num = 46;
    ramulator::core_configs[core_type_t::SILVER].l2_cache_config.access_energy = 29.70 * 2.00;

    ramulator::core_configs[core_type_t::GOLD].l2_cache_config.size = 1 << 17;
    ramulator::core_configs[core_type_t::GOLD].l2_cache_config.assoc = 1 << 2;
    ramulator::core_configs[core_type_t::GOLD].l2_cache_config.blocksz = 1 << 6;
    ramulator::core_configs[core_type_t::GOLD].l2_cache_config.mshr_num = 46;
    ramulator::core_configs[core_type_t::GOLD].l2_cache_config.access_energy = 29.70 * 2.00;

    ramulator::core_configs[core_type_t::PRIME].l2_cache_config.size = 1 << 19;
    // ramulator::core_configs[core_type_t::PRIME].l2_cache_config.size = 1 << 19;
    ramulator::core_configs[core_type_t::PRIME].l2_cache_config.assoc = 1 << 3;
    // ramulator::core_configs[core_type_t::PRIME].l2_cache_config.assoc = 1 << 3;
    ramulator::core_configs[core_type_t::PRIME].l2_cache_config.blocksz = 1 << 6;
    ramulator::core_configs[core_type_t::PRIME].l2_cache_config.mshr_num = 46;
    ramulator::core_configs[core_type_t::PRIME].l2_cache_config.access_energy = 29.70 * 2.00;

    ramulator::core_configs[core_type_t::SILVER].ipc = 1;
    ramulator::core_configs[core_type_t::GOLD].ipc = 3;
    ramulator::core_configs[core_type_t::PRIME].ipc = 3;

    ramulator::core_configs[core_type_t::SILVER].MVE_SA_num = 8;
    ramulator::core_configs[core_type_t::GOLD].MVE_SA_num = 16;
    ramulator::core_configs[core_type_t::PRIME].MVE_SA_num = 32;

    ramulator::core_configs[core_type_t::SILVER].out_of_order = false;
    ramulator::core_configs[core_type_t::GOLD].out_of_order = true;
    ramulator::core_configs[core_type_t::PRIME].out_of_order = true;
}

std::ofstream ramulator::op_trace;

template <typename T>
void run_dramtrace(const Config &configs, Memory<T, Controller> &memory, const std::vector<const char *> &files) {

    /* initialize DRAM trace */
    // assuming only one file
    assert(files.size() == 1);
    Trace trace(files);

    /* run simulation */
    bool stall = false, end = false;
    int reads = 0, writes = 0, clks = 0;
    long addr = 0;
    Request::Type type = Request::Type::READ;
    map<int, int> latencies;
    auto read_complete = [&latencies](Request &r) { latencies[r.depart - r.arrive]++; };

    Request req(addr, type, read_complete);

    while (!end || memory.pending_requests()) {
        if (!end && !stall) {
            end = !trace.get_dramtrace_request(addr, type);
        }

        if (!end) {
            req.addr = addr;
            req.type = type;
            stall = !memory.send(req);
            if (!stall) {
                if (type == Request::Type::READ)
                    reads++;
                else if (type == Request::Type::WRITE)
                    writes++;
            }
        } else {
            memory.set_high_writeq_watermark(0.0f); // make sure that all write requests in the
                                                    // write queue are drained
        }

        memory.tick();
        clks++;
        Stats::curTick++; // memory clock, global, for Statistics
    }
    // This a workaround for statistics set only initially lost in the end
    memory.finish();
    Stats::statlist.printall();
}

template <typename T>
void run_cputrace(const Config &configs, Memory<T, Controller> &memory, const std::vector<const char *> &files) {
    int cpu_tick = configs.get_cpu_tick();
    int mem_tick = configs.get_mem_tick();
    auto send = bind(&Memory<T, Controller>::send, &memory, placeholders::_1);
    declare_configuration(configs);
    Processor proc(configs, files, send, memory);

    long warmup_insts = configs.get_warmup_insts();
    bool is_warming_up = (warmup_insts != 0);

    for (long i = 0; is_warming_up; i++) {
        proc.tick();
        Stats::curTick++;
        if (i % cpu_tick == (cpu_tick - 1))
            for (int j = 0; j < mem_tick; j++)
                memory.tick();

        is_warming_up = false;
        for (int c = 0; c < proc.cores.size(); c++) {
            if (proc.cores[c]->get_insts() < warmup_insts)
                is_warming_up = true;
        }

        if (is_warming_up && proc.has_reached_limit()) {
            printf("WARNING: The end of the input trace file was reached during warmup. "
                   "Consider changing warmup_insts in the config file. \n");
            break;
        }
    }

    warmup_complete = true;
    printf("Warmup complete! Resetting stats...\n");
    Stats::reset_stats();
    proc.reset_stats();
    assert(proc.get_insts() == 0);

    printf("Starting the simulation...\n");

    int tick_mult = cpu_tick * mem_tick;
    for (long i = 0;; i++) {
        if (((i % tick_mult) % mem_tick) == 0) { // When the CPU is ticked cpu_tick times,
            // the memory controller should be ticked mem_tick times
            proc.tick();
            Stats::curTick++; // processor clock, global, for Statistics

            if (configs.calc_weighted_speedup()) {
                if (proc.has_reached_limit()) {
                    break;
                }
            } else {
                if (configs.is_early_exit()) {
                    if (proc.finished())
                        break;
                } else {
                    if (proc.finished() && (memory.pending_requests() == 0))
                        break;
                }
            }
        }

        if (((i % tick_mult) % cpu_tick) == 0) // TODO_hasan: Better if the processor ticks the memory controller
            memory.tick();
    }
    // This a workaround for statistics set only initially lost in the end
    memory.finish();
    Stats::statlist.printall();
}

template <typename T>
void run_MVEtrace(const Config &configs, Memory<T, Controller> &memory, const std::vector<const char *> &files) {
    int cpu_tick = configs.get_cpu_tick();
    int mem_tick = configs.get_mem_tick();
    auto send = bind(&Memory<T, Controller>::send, &memory, placeholders::_1);
    declare_configuration(configs);
    Processor proc(configs, files, send, memory);

    bool is_warming_up = configs.is_warming_up();

    for (long i = 0; is_warming_up; i++) {
        proc.tick();
        Stats::curTick++;
        if (i % cpu_tick == (cpu_tick - 1))
            for (int j = 0; j < mem_tick; j++)
                memory.tick();

        is_warming_up = false;
        for (int c = 0; c < proc.cores.size(); c++) {
            if (proc.cores[c]->is_warmed_up() == false) {
                is_warming_up = true;
                break;
            }
        }

        if (is_warming_up && proc.has_reached_limit()) {
            printf("WARNING: The end of the input trace file was reached during warmup. "
                   "Consider changing warmup_insts in the config file. \n");
            break;
        }
    }

    warmup_complete = true;
    printf("Warmup complete! Resetting stats...\n");
    Stats::curTick = 0;
    Stats::reset_stats();
    proc.reset_state();
    assert(proc.get_insts() == 0);

    printf("Starting the simulation...\n");

    int tick_mult = cpu_tick * mem_tick;
    for (long i = 0;; i++) {
        if (((i % tick_mult) % mem_tick) == 0) { // When the CPU is ticked cpu_tick times,
            // the memory controller should be ticked mem_tick times
            proc.tick();
            Stats::curTick++; // processor clock, global, for Statistics

            if (configs.calc_weighted_speedup()) {
                if (proc.has_reached_limit()) {
                    break;
                }
            } else {
                if (configs.is_early_exit()) {
                    if (proc.finished())
                        break;
                } else {
                    if (proc.finished() && (memory.pending_requests() == 0))
                        break;
                }
            }
        }

        if (((i % tick_mult) % cpu_tick) == 0) // TODO_hasan: Better if the processor ticks the memory controller
            memory.tick();
    }
    // This a workaround for statistics set only initially lost in the end
    memory.finish();
    Stats::statlist.printall();
    proc.reset_state();
}

ramulator::Cache *dc_llc;
ramulator::Cache *dc_l2;
std::shared_ptr<CacheSystem> dc_cachesys;
vector<vector<Request>> dc_block_tosend_instrs[8];
vector<Request> dc_block_sent_requests[8];
int dc_current_block_num[8];
bool trace_finished = false;
Trace *dc_trace;
bool read_new_block = false;
int id = 0;
int total_access = 0;
long dc_clks = 0;

void dc_receive(Request &req) {
    hint("DC received %s\n", req.c_str());
    dc_llc->callback(req);

    // Removing corresponding DC sent requests
    for (int block_idx = 0; block_idx < 8; block_idx++) {
        auto req_it = dc_block_sent_requests[block_idx].begin();
        bool hit = false;
        while (req_it != dc_block_sent_requests[block_idx].end()) {
            if (dc_l2->align(req.addr) == dc_l2->align(req_it->addr)) {
                hint("Block [%d]: %s hitted with %s, removing from sent list\n", block_idx, req.c_str(), req_it->c_str());
                op_trace << dc_clks << " " << block_idx << " Received 0x" << std::hex << req.addr << std::dec << endl;
                req_it = dc_block_sent_requests[block_idx].erase(req_it);
                hit = true;
            } else {
                ++req_it;
            }
        }
        if (hit && (dc_block_sent_requests[block_idx].size() == 0) && (dc_block_tosend_instrs[block_idx][0].size() == 0)) {
            dc_block_tosend_instrs[block_idx].erase(dc_block_tosend_instrs[block_idx].begin());
            op_trace << dc_clks << " " << block_idx << " Finished " << dc_current_block_num[block_idx] << endl;
        }
    }
}

void get_new_block(int block) {
    long addr = 0;
    Request::Type type = Request::Type::READ;
    Request::Type RW_type = Request::Type::MAX;
    assert(trace_finished == false);
    while (trace_finished == false) {
        trace_finished = !dc_trace->get_dramtrace_request(addr, type);
        if (trace_finished == false) {
            if (type == Request::Type::DC_BLOCK) {
                if (read_new_block) {
                    dc_current_block_num[block] = addr - 1;
                    op_trace << dc_clks << " " << block << " Started " << dc_current_block_num[block] << endl;
                    break;
                }
                read_new_block = true;
            } else if (type == Request::Type::READ) {
                int curr_size = dc_block_tosend_instrs[block].size();
                if (curr_size != 0) {
                    if (dc_block_tosend_instrs[block][curr_size - 1].size() == 0) {
                        dc_block_tosend_instrs[block].erase(dc_block_tosend_instrs[block].begin() + (curr_size - 1));
                    }
                }
                dc_block_tosend_instrs[block].push_back(vector<Request>());
                RW_type = Request::Type::READ;
            } else if (type == Request::Type::WRITE) {
                int curr_size = dc_block_tosend_instrs[block].size();
                if (curr_size != 0) {
                    if (dc_block_tosend_instrs[block][curr_size - 1].size() == 0) {
                        dc_block_tosend_instrs[block].erase(dc_block_tosend_instrs[block].begin() + (curr_size - 1));
                    }
                }
                dc_block_tosend_instrs[block].push_back(vector<Request>());
                RW_type = Request::Type::WRITE;
            } else {
                assert(type == Request::Type::MAX);
                assert((RW_type == Request::Type::READ) || (RW_type == Request::Type::WRITE));
                Request req(addr, RW_type);
                req.coreid = 0;
                req.callback = dc_receive;
                req.dc_blockid = block;
                req.addr = addr;
                req.type = RW_type;
                req.reqid = id++;
                dc_block_tosend_instrs[block][dc_block_tosend_instrs[block].size() - 1].push_back(req);
                total_access++;
            }
        }
    }
}

void dc_blocks_clock(int block) {

    if (dc_block_tosend_instrs[block].size() == 0) {
        if (trace_finished)
            return;
        get_new_block(block);
    }
    hint("Block [%d]: Clocking...\n", block);
    while (dc_block_tosend_instrs[block][0].size() > 0) {
        Request req = dc_block_tosend_instrs[block][0][0];
        assert((req.type == Request::Type::READ) || (req.type == Request::Type::WRITE));
        if (dc_l2->should_send(req) == false) {
            hint("Block [%d]: Mem addr should not be sent (%s)\n", block, req.c_str());
            break;
        } else {
            if (dc_l2->send(req) == false) {
                hint("Block [%d]: Mem addr failed to be sent (%s)\n", block, req.c_str());
                op_trace << dc_clks << " " << block << " Failed  0x" << std::hex << req.addr << std::dec << endl;
                break;
            } else {
                hint("Block [%d]: Mem addr sent, removed from tosend and added to sent (%s)\n", block, req.c_str());
                dc_block_sent_requests[block].push_back(req);
                dc_block_tosend_instrs[block][0].erase(dc_block_tosend_instrs[block][0].begin());
                op_trace << dc_clks << " " << block << " Sent 0x" << std::hex << req.addr << std::dec << endl;
            }
        }
    }
}

void dc_blocks_clock_all() {
    hint("Clocking all blocks\n");
    int block_start = rand() % 8;
    for (int block_offset = 0; block_offset < 8; block_offset++) {
        int block_idx = (block_start + block_offset) % 8;
        dc_blocks_clock(block_idx);
    }
}

template <typename T>
void run_dctrace(const Config &configs, Memory<T, Controller> &memory, const std::vector<const char *> &files) {
    dc_trace = new Trace(files);

    int cpu_tick = configs.get_cpu_tick();
    int mem_tick = configs.get_mem_tick();

    auto send_memory = bind(&Memory<T, Controller>::send, &memory, placeholders::_1);
    declare_configuration(configs);

    dc_cachesys = std::shared_ptr<CacheSystem>(new CacheSystem(configs, send_memory));
    dc_llc = new ramulator::Cache(ramulator::l3_size, ramulator::l3_assoc, ramulator::l3_blocksz, ramulator::mshr_per_bank * configs.get_core_num(), l3_access_energy, Cache::Level::L3, dc_cachesys, l3_MVE_SA_num);

    dc_l2 = new ramulator::Cache(
        core_configs[ramulator::core_type_t::PRIME].l2_cache_config.size,
        core_configs[ramulator::core_type_t::PRIME].l2_cache_config.assoc,
        core_configs[ramulator::core_type_t::PRIME].l2_cache_config.blocksz,
        core_configs[ramulator::core_type_t::PRIME].l2_cache_config.mshr_num,
        core_configs[ramulator::core_type_t::PRIME].l2_cache_config.access_energy,
        Cache::Level::L2,
        dc_cachesys,
        core_configs[ramulator::core_type_t::PRIME].MVE_SA_num,
        0);
    dc_l2->concatlower(dc_llc);

    /* run simulation */
    int clks = 0;
    int tick_mult = cpu_tick * mem_tick;
    long i = 0;
    bool sim_finished = false;
    while (sim_finished == false) {
        if (trace_finished && dc_cachesys->finished() && dc_l2->finished() && dc_llc->finished() && (!memory.pending_requests())) {
            sim_finished = true;
            for (int block_idx = 0; block_idx < 8; block_idx++) {
                if (dc_block_tosend_instrs[block_idx].size() != 0) {
                    sim_finished = false;
                    break;
                }
                if (dc_block_sent_requests[block_idx].size() != 0) {
                    sim_finished = false;
                    break;
                }
            }
        }

        if (((i % tick_mult) % mem_tick) == 0) { // When the CPU is ticked cpu_tick times,
            dc_blocks_clock_all();
            dc_l2->tick();
            dc_cachesys->tick();
            if (dc_clks % 1000000 == 0) {
                printf("DC heartbeat, cycles: %ld \n", dc_clks);
            }
            dc_clks += 1;
        }
        if (((i % tick_mult) % cpu_tick) == 0) {
            memory.tick();
        }
        clks++;
        Stats::curTick++;
        i++;
    }
    // This a workaround for statistics set only initially lost in the end
    memory.finish();
    Stats::statlist.printall();
    printf("finished %d accesses in %ld clks\n", total_access, dc_clks);
}

template <typename T>
void start_run(const Config &configs, T *spec, const vector<const char *> &files) {
    // initiate controller and memory
    int C = configs.get_channels(), R = configs.get_ranks();
    // Check and Set channel, rank number
    spec->set_channel_number(C);
    spec->set_rank_number(R);
    std::vector<Controller<T> *> ctrls;
    int prev_children = 0;
    for (int c = 0; c < C; c++) {
        DRAM<T> *channel = new DRAM<T>(spec, T::Level::Channel);
        channel->id = c;
        channel->regStats("");
        prev_children = channel->set_index(prev_children);
        Controller<T> *ctrl = new Controller<T>(configs, channel);
        ctrls.push_back(ctrl);
    }
    Memory<T, Controller> memory(configs, ctrls);

    assert(files.size() != 0);
    if (configs["trace_type"] == "CPU") {
        run_cputrace(configs, memory, files);
    } else if (configs["trace_type"] == "DRAM") {
        run_dramtrace(configs, memory, files);
    } else if (configs["trace_type"] == "MVE") {
        run_MVEtrace(configs, memory, files);
    } else if (configs["trace_type"] == "DC") {
        run_dctrace(configs, memory, files);
    }
}

int main(int argc, const char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <configs-file> --mode=cpu,dram,MVE [--warmup] [--core=<core-num> <core1-type> ...] [--stats <filename>] <trace-filename1> <trace-filename2>\n"
               "Example: %s ramulator-configs.cfg --mode=MVE --core=1 gold cpu.trace cpu.trace\n",
               argv[0], argv[0]);
        return 0;
    }

    Config configs(argv[1]);

    const std::string &standard = configs["standard"];
    assert(standard != "" || "DRAM standard should be specified.");

    const char *trace_type = strstr(argv[2], "=");
    trace_type++;
    if (strcmp(trace_type, "cpu") == 0) {
        configs.add("trace_type", "CPU");
    } else if (strcmp(trace_type, "dram") == 0) {
        configs.add("trace_type", "DRAM");
    } else if (strcmp(trace_type, "MVE") == 0) {
        configs.add("trace_type", "MVE");
        assert((configs.has_core_caches() && configs.has_l3_cache()) || "MVE mode need \"all\" cache levels");
    } else if (strcmp(trace_type, "dc") == 0) {
        configs.add("trace_type", "DC");
    } else {
        printf("invalid trace type: %s\n", trace_type);
        assert(false);
    }

    int trace_start = 3;

    if (strstr(argv[3], "--warmup") != nullptr) {
        configs.set_warming_up();
        trace_start++;
    }

    int core_num = 0;
    std::vector<std::string> core_types;
    const char *core_num_str = strstr(argv[trace_start], "--core=");
    if (core_num_str != nullptr) {
        core_num_str = strstr(argv[trace_start], "=");
        core_num_str++;
        core_num = stoi(core_num_str);
        if (core_num < 1 || core_num > 16) {
            printf("core num must be >= 1 and <= 16\n");
            assert(false);
        }
        trace_start++;
        for (int core_idx = 0; core_idx < core_num; core_idx++) {
            const char *core_type = argv[trace_start];
            trace_start++;
            assert((strcmp(core_type, "silver") == 0) || (strcmp(core_type, "gold") == 0) || (strcmp(core_type, "prime") == 0));
            core_types.push_back(std::string(core_type));
        }
    }

    string stats_out;
    if (strcmp(argv[trace_start], "--stats") == 0) {
        Stats::statlist.output(argv[trace_start + 1]);
        stats_out = argv[trace_start + 1];
        trace_start += 2;
    } else {
        Stats::statlist.output(standard + ".stats");
        stats_out = standard + string(".stats");
    }

    op_trace.open(stats_out + ".map");
    if (!op_trace.is_open()) {
        printf("Could not open %s.map\n", (stats_out + ".map").c_str());
        return -1;
    }

    // A separate file defines mapping for easy config.
    if (strcmp(argv[trace_start], "--mapping") == 0) {
        configs.add("mapping", argv[trace_start + 1]);
        trace_start += 2;
    } else {
        configs.add("mapping", "defaultmapping");
    }

    std::vector<const char *> files(&argv[trace_start], &argv[argc]);
    if (core_num != 0) {
        if (argc - trace_start < core_num) {
            printf("number of trace files must be >= core num\n");
            assert(false);
        }
        configs.set_core_num(core_num);
        configs.set_core_types(core_types);
    } else {
        configs.set_core_num(argc - trace_start);
    }

    if (standard == "DDR3") {
        DDR3 *ddr3 = new DDR3(configs["org"], configs["speed"]);
        start_run(configs, ddr3, files);
    } else if (standard == "DDR4") {
        DDR4 *ddr4 = new DDR4(configs["org"], configs["speed"]);
        start_run(configs, ddr4, files);
    } else if (standard == "SALP-MASA") {
        SALP *salp8 = new SALP(configs["org"], configs["speed"], "SALP-MASA", configs.get_subarrays());
        start_run(configs, salp8, files);
    } else if (standard == "LPDDR3") {
        LPDDR3 *lpddr3 = new LPDDR3(configs["org"], configs["speed"]);
        start_run(configs, lpddr3, files);
    } else if (standard == "LPDDR4") {
        // total cap: 2GB, 1/2 of others
        LPDDR4 *lpddr4 = new LPDDR4(configs["org"], configs["speed"]);
        start_run(configs, lpddr4, files);
    } else if (standard == "GDDR5") {
        GDDR5 *gddr5 = new GDDR5(configs["org"], configs["speed"]);
        start_run(configs, gddr5, files);
    } else if (standard == "HBM") {
        HBM *hbm = new HBM(configs["org"], configs["speed"]);
        start_run(configs, hbm, files);
    } else if (standard == "WideIO") {
        // total cap: 1GB, 1/4 of others
        WideIO *wio = new WideIO(configs["org"], configs["speed"]);
        start_run(configs, wio, files);
    } else if (standard == "WideIO2") {
        // total cap: 2GB, 1/2 of others
        WideIO2 *wio2 = new WideIO2(configs["org"], configs["speed"], configs.get_channels());
        wio2->channel_width *= 2;
        start_run(configs, wio2, files);
    } else if (standard == "STTMRAM") {
        STTMRAM *sttmram = new STTMRAM(configs["org"], configs["speed"]);
        start_run(configs, sttmram, files);
    } else if (standard == "PCM") {
        PCM *pcm = new PCM(configs["org"], configs["speed"]);
        start_run(configs, pcm, files);
    }
    // Various refresh mechanisms
    else if (standard == "DSARP") {
        DSARP *dsddr3_dsarp = new DSARP(configs["org"], configs["speed"], DSARP::Type::DSARP, configs.get_subarrays());
        start_run(configs, dsddr3_dsarp, files);
    } else if (standard == "ALDRAM") {
        ALDRAM *aldram = new ALDRAM(configs["org"], configs["speed"]);
        start_run(configs, aldram, files);
    } else if (standard == "TLDRAM") {
        TLDRAM *tldram = new TLDRAM(configs["org"], configs["speed"], configs.get_subarrays());
        start_run(configs, tldram, files);
    }

    op_trace.close();
    printf("Simulation done. Statistics written to %s\n", stats_out.c_str());

    return 0;
}

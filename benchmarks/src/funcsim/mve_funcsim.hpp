
#ifndef __MVE_INCLUDE_HPP__
#define __MVE_INCLUDE_HPP__

#include <fstream>
#include <stdbool.h>
#include <string>
#include <vector>

// #define DEBUG

using namespace std;

#include "mve_variables.hpp"

#define MVE_COMPARE

class operation {
public:
    void const *src_address;
    void const *dst_address;
    double constant_input;
    double constant_output;
    int first_operand;
    int second_operand;
    int third_operand;
    int result_operand;
    std::string type;
    operation(std::string t, double c_i, double c_o, int a, int b, int c, int o, void const *m, void const *n);
};

class mve {
    static int float_output_id;
    static bool analysis_enabled;
    static int flushed_registers;
    static int freed_registers;
    static long registers_size;

public:
#ifdef MVE_COMPARE
    static std::vector<__mdv_var> registers;
#endif
    static std::vector<operation> operations;
    static std::ofstream graph_f;
    static int LANE_NUM;
    static __dim_var dims[4];
    static bool *mask;
    static int dim_count;
    mve();
    static __int64_t new_register(__mdv_var o);

    static void new_operation(std::string t);                                                      // free pr
    static void new_operation(std::string t, int count);                                           // set config
    static void new_operation(std::string t, int dim, int config);                                 // set config
    static __int64_t new_operation(std::string t, __int64_t a, __int64_t b, __mdv_var o);          // add, mul, min, max, shiftr
    static void new_operation(std::string t, __int64_t a, __int64_t b);                            // cmpeq, cmpgte
    static void new_operation(std::string t, __int64_t a);                                         // reduction
    static __int64_t new_operation(std::string t, void const *m, __int64_t a, __mdv_var o);        // dict
    static __int64_t new_operation(std::string t, void const *m, __mdv_var o, __vidx_var stride);  // load
    static __int64_t new_operation(std::string t, void const **m, __mdv_var o, __vidx_var stride); // loadr
    static void new_operation(std::string t, __int64_t a, void const *n, __vidx_var stride);       // store
    static void new_operation(std::string t, __int64_t a, void const **n, __vidx_var stride);      // storer
    static __int64_t new_operation(std::string t, double c_i, __mdv_var o);                        // set1
    static __int64_t new_operation(std::string t, __int64_t a, double c_i, __mdv_var o);           // shifti
    static __int64_t new_operation(std::string t, __int64_t a, __mdv_var o);                       // convert

    static int *stride_evaluator(__vidx_var rstride, bool load);
    static __mdv_var get_value(int id);
    static void initializer(char *exp_name, int LANE_NUM);
    static void init_dims();
    static void set_mask();
    static void finisher();
    static void flusher();
    static void start_analysis(void);
    static void end_analysis(void);
    static void dump_graph();
    static void free_register();
    static void printer_reg(__int64_t dst, __int64_t src1, __int64_t src2, std::string t, int config, int value);
    static void printer_mem(__int64_t dst, const void *src1, __int64_t src2, std::string t, int value1, int value2, int value3, int value4);
};

#endif

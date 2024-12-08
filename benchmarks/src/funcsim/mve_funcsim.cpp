
#include "mve_funcsim.hpp"
#include <asm-generic/errno.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <sys/stat.h>

#ifdef MVE_COMPARE
std::vector<__mdv_var> mve::registers;
#endif

std::vector<operation> mve::operations;

int mve::float_output_id = 0;
bool mve::analysis_enabled = false;
int mve::flushed_registers = 0;
int mve::LANE_NUM = 0;
__dim_var mve::dims[4];
int mve::dim_count = 4;
bool *mve::mask = NULL;
int mve::freed_registers = 0;
long mve::registers_size;

std::ofstream mve::graph_f;

operation::operation(std::string t, double c_i, double c_o, int a, int b, int c, int o, void const *m, void const *n) {
    src_address = m;
    dst_address = n;
    constant_input = c_i;
    constant_output = c_o;
    first_operand = a;
    second_operand = b;
    third_operand = c;
    result_operand = o;
    type = t;
}

__int64_t mve::new_register(__mdv_var o) {
    int id = mve::registers_size + mve::flushed_registers;
#ifdef MVE_COMPARE
    mve::registers.push_back(o);
#endif
    mve::registers_size += 1;
    return id;
}

void mve::printer_reg(__int64_t dst, __int64_t src1, __int64_t src2, std::string t, int config, int value) {
    graph_f << t << " " << dst << " " << src1 << " " << src2 << " " << config << " " << value << endl;
}

void mve::printer_mem(__int64_t dst, const void *src1, __int64_t src2, std::string t, int value1, int value2, int value3, int value4) {
    graph_f << t << " " << dst << " " << src1 << " " << src2 << " " << value1 << " " << value2 << " " << value3 << " " << value4 << endl;
}

void mve::new_operation(std::string t) { // free pr
    printer_reg(-1, -1, -1, t, -1, -1);
}

void mve::new_operation(std::string t, int value) { // set dim count
    printer_reg(-1, -1, -1, t, -1, value);
    assert(t.find("set_dim_count") != string::npos);
    assert((value >= 1) && (value <= 4));
    mve::dim_count = value;
}

void mve::new_operation(std::string t, int config, int value) { // set config
    assert(config >= 0 && config < 4);
    assert(config < mve::dim_count);
    printer_reg(-1, -1, -1, t, config, value);
    if (t.find("length") != string::npos) {
        mve::dims[config].length = value;
        if (mve::dims[config].mask != NULL) {
            delete[] mve::dims[config].mask;
            mve::dims[config].mask = NULL;
        }
        mve::dims[config].mask = new bool[value];
        for (int element = 0; element < value; element++) {
            mve::dims[config].mask[element] = true;
        }
    } else if (t.find("element") != string::npos) {
        bool val = true;
        if (t.find("unset") != string::npos) {
            val = false;
        }
        if (t.find("all") != string::npos) {
            for (int element = 0; element < mve::dims[config].length; element++) {
                mve::dims[config].mask[element] = val;
            }
        } else if (t.find("only") != string::npos) {
            for (int element = 0; element < mve::dims[config].length; element++) {
                mve::dims[config].mask[element] = !val;
            }
            mve::dims[config].mask[value] = val;
        } else if (t.find("active") != string::npos) {
            mve::dims[config].mask[value] = val;
        } else if (t.find("half") != string::npos) {
            assert((mve::dims[config].length % 2) == 0);
            int min_element = 0;
            int max_element = mve::dims[config].length;
            if (t.find("first") != string::npos) {
                max_element /= 2;
            } else if (t.find("second") != string::npos) {
                min_element = max_element / 2;
            } else {
                assert(false);
            }
            for (int element = min_element; element < max_element; element++) {
                mve::dims[config].mask[element] = !val;
            }
        } else {
            assert(false);
        }
    } else if (t.find("load_stride") != string::npos) {
        mve::dims[config].load_stride = value;
    } else if (t.find("store_stride") != string::npos) {
        mve::dims[config].store_stride = value;
    } else {
        assert(false);
    }
}

__int64_t mve::new_operation(std::string t, __int64_t a, __int64_t b, __mdv_var o) { // add, mul, min, max, shiftr, assign
    int id = new_register(o);
    if (analysis_enabled) {
        printer_reg(id, a, b, t, -1, -1);
    }
    return id;
}

void mve::new_operation(std::string t, __int64_t a, __int64_t b) { // cmp
    if (analysis_enabled) {
        printer_reg(-1, a, b, t, -1, -1);
    }
}

void mve::new_operation(std::string t, __int64_t a) { // reduction
    if (analysis_enabled) {
        printer_reg(-1, a, -1, t, -1, -1);
    }
}

__int64_t mve::new_operation(std::string t, void const *m, __int64_t a, __mdv_var o) { // dict
    int id = new_register(o);
    if (analysis_enabled) {
        printer_mem(id, m, a, t, -1, -1, -1, -1);
    }
    return id;
}

__int64_t mve::new_operation(std::string t, void const *m, __mdv_var o, __vidx_var stride) { // load
    int id = new_register(o);
    if (analysis_enabled) {
        printer_mem(id, m, -1, t, stride[3], stride[2], stride[1], stride[0]);
    }
    return id;
}

__int64_t mve::new_operation(std::string t, void const **m, __mdv_var o, __vidx_var stride) { // loadr
    int id = new_register(o);
    if (analysis_enabled) {
        printer_mem(id, m, -1, t, stride[3], stride[2], stride[1], stride[0]);

        for (int i = 0; i < mve::dims[mve::dim_count - 1].length; i++) {
            graph_f << m[i] << " ";
        }
        graph_f << endl;
    }
    return id;
}

void mve::new_operation(std::string t, __int64_t a, void const *n, __vidx_var stride) { // store
    if (analysis_enabled) {
        printer_mem(-1, n, a, t, stride[3], stride[2], stride[1], stride[0]);
    }
}

void mve::new_operation(std::string t, __int64_t a, void const **n, __vidx_var stride) { // storer
    if (analysis_enabled) {
        printer_mem(-1, n, a, t, stride[3], stride[2], stride[1], stride[0]);

        for (int i = 0; i < mve::dims[mve::dim_count - 1].length; i++) {
            graph_f << n[i] << " ";
        }
        graph_f << endl;
    }
}

__int64_t mve::new_operation(std::string t, double c_i, __mdv_var o) { // set1
    int id = new_register(o);
    if (analysis_enabled) {
        printer_reg(id, -1, -1, t, -1, -1);
    }
    return id;
}

__int64_t mve::new_operation(std::string t, __int64_t a, double c_i, __mdv_var o) { // shifti
    int id = new_register(o);
    if (analysis_enabled) {
        printer_reg(id, a, -1, t, -1, -1);
    }
    return id;
}

__int64_t mve::new_operation(std::string t, __int64_t a, __mdv_var o) { // convert and copy
    int id = new_register(o);
    if (analysis_enabled) {
        printer_reg(id, a, -1, t, -1, -1);
    }
    return id;
}

__mdv_var mve::get_value(int id) {
    if (id >= mve::registers_size + mve::flushed_registers) {
        cout << "Error: requesting register ID#" << id << " while there are " << mve::registers_size << " registers!" << endl;
        assert(false);
    } else if (id < mve::flushed_registers) {
        // cout << "Warning: requesting register ID#" << id << " is between " << mve::flushed_registers << " flsuhed registers! Returning 0!" << endl;
        __mdv_var temp(mve::dims);
        return temp;
    }
#ifdef MVE_COMPARE
    return registers[id - mve::flushed_registers];
#else
    __mdv_var temp(mve::dims);
    return temp;
#endif
}

inline bool file_exists(const std::string &name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

void mve::init_dims() {
    graph_f << "mve_init_dims" << endl;
    for (int dim = 0; dim < 4; dim++) {
        mve::dims[dim].length = 1;
        if (mve::dims[dim].mask != NULL) {
            delete[] mve::dims[dim].mask;
            mve::dims[dim].mask = NULL;
        }
        mve::dims[dim].mask = new bool[1];
        mve::dims[dim].mask[0] = true;
        mve::dims[dim].load_stride = -1;
        mve::dims[dim].store_stride = -1;
    }
    mve::set_mask();
    mve::dims[0].length = LANE_NUM;
    delete[] mve::dims[0].mask;
    mve::dims[0].mask = new bool[LANE_NUM];
    for (int element = 0; element < LANE_NUM; element++) {
        mve::dims[0].mask[element] = true;
    }
    mve::dim_count = 4;
}

void mve::initializer(char *exp_name, int LANE_NUM) {
    string graph_name;
    string full_name;
    graph_name = (string)exp_name;
    full_name = graph_name + ".dfg";
    if (file_exists(full_name)) {
        int dup_num;
        for (dup_num = 1; dup_num != -1; dup_num++) {
            full_name = graph_name + "_(" + to_string(dup_num) + ").dfg";
            if (file_exists(full_name)) {
                continue;
            }
            break;
        }
    }
    mve::LANE_NUM = LANE_NUM;
    init_dims();
    mve::flushed_registers += mve::registers_size;
#ifdef MVE_COMPARE
    mve::registers.clear();
#endif
    mve::registers_size = 0;
    mve::graph_f.open(full_name);
    cout << "file " << full_name << endl;
    mve::start_analysis();
}

void mve::set_mask() {
    if (mve::mask != NULL) {
        delete[] mve::mask;
        mve::mask = NULL;
    }
    mve::mask = new bool[mve::LANE_NUM];
    for (int i = 0; i < LANE_NUM; i++) {
        mve::mask[i] = true;
    }
}

void mve::finisher() {
    cout << "Finishing MVE computation" << endl;
    mve::end_analysis();
    if (mve::graph_f.is_open()) {
        mve::graph_f.close();
    } else {
        cout << "Warning: closing an already closed file!" << endl;
    }
    // if (mve::freed_registers != (mve::registers_size + flushed_registers)) {
    //     cout << "Error: freed registers (" << mve::freed_registers << ") != register count (" << mve::registers_size << ") + flushed registers (" << flushed_registers << ")\n";
    //     assert(false);
    // }
    mve::freed_registers = 0;
#ifdef MVE_COMPARE
    mve::registers.clear();
#endif
    mve::registers_size = 0;
    mve::operations.clear();
    mve::float_output_id = 0;
    mve::analysis_enabled = false;
    mve::flushed_registers = 0;
    delete[] mve::mask;
    mve::mask = NULL;
    for (int dim = 0; dim < 4; dim++) {
        if (mve::dims[dim].mask != NULL) {
            delete[] mve::dims[dim].mask;
            mve::dims[dim].mask = NULL;
        }
    }
}

void mve::flusher() {
    mve::flushed_registers += mve::registers_size;
#ifdef MVE_COMPARE
    mve::registers.clear();
#endif
    mve::registers_size = 0;
    graph_f << "flushed" << endl;
}

void mve::start_analysis(void) { analysis_enabled = true; }

void mve::end_analysis(void) { analysis_enabled = false; }

void mve::free_register(void) {
    mve::freed_registers += 1;
}

int *mve::stride_evaluator(__vidx_var rstride, bool load) {
    int *lstride = new __vidx_var;
    for (int dim = 0; dim < 4; dim++) {
        if (dim >= mve::dim_count)
            assert(rstride[dim] == 0);
        switch (rstride[dim]) {
        case 3:
            if (load) {
                assert(mve::dims[dim].load_stride >= 0);
                lstride[dim] = mve::dims[dim].load_stride;
            } else {
                assert(mve::dims[dim].store_stride >= 0);
                lstride[dim] = mve::dims[dim].store_stride;
            }
            break;
        case 2:
            if (dim == 0) {
                lstride[dim] = 1;
            } else {
                lstride[dim] = mve::dims[dim - 1].length * lstride[dim - 1];
            }
            break;
        default:
            lstride[dim] = rstride[dim];
        }
    }
    return lstride;
}

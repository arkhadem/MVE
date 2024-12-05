#include "expand_palette.hpp"

#include "libpng.hpp"

#include "benchmark.hpp"

#include "init.hpp"

int expand_palette_init(size_t cache_size,
                        int LANE_NUM,
                        config_t *&config,
                        input_t **&input,
                        output_t **&output) {

    expand_palette_config_t *expand_palette_config = (expand_palette_config_t *)config;
    expand_palette_input_t **expand_palette_input = (expand_palette_input_t **)input;
    expand_palette_output_t **expand_palette_output = (expand_palette_output_t **)output;

    // configuration
    int rows = 16;
    int columns = 1024;

    init_1D<expand_palette_config_t>(1, expand_palette_config);
    expand_palette_config->num_rows = rows;
    expand_palette_config->num_cols = columns;

    // in/output versions
    size_t input_size = (rows * columns + 7 * 256) * sizeof(png_byte);
    input_size += (3 * 256) * sizeof(png_uint_32);
    size_t output_size = (rows * columns * 4) * sizeof(png_byte);
    int count = cache_size / (input_size + output_size) + 1;

    // initializing in/output versions
    init_1D<expand_palette_input_t *>(count, expand_palette_input);
    init_1D<expand_palette_output_t *>(count, expand_palette_output);

    // initializing individual versions
    for (int i = 0; i < count; i++) {
        init_1D<expand_palette_input_t>(1, expand_palette_input[i]);
        init_1D<expand_palette_output_t>(1, expand_palette_output[i]);

        random_init_2D<png_byte>(rows, columns, expand_palette_input[i]->input_buf);
        random_init_1D<png_uint_32>(256, expand_palette_input[i]->riffled_palette);
        random_init_1D<png_byte>(3 * 256, expand_palette_input[i]->rgb_palette);
        init_1D<png_byte>(256, expand_palette_input[i]->r_palette);
        init_1D<png_byte>(256, expand_palette_input[i]->g_palette);
        init_1D<png_byte>(256, expand_palette_input[i]->b_palette);
        random_init_1D<png_byte>(256, expand_palette_input[i]->a_palette);
        for (int j = 0; j < 256; j++) {
            expand_palette_input[i]->r_palette[j] = expand_palette_input[i]->rgb_palette[3 * j + 0];
            expand_palette_input[i]->g_palette[j] = expand_palette_input[i]->rgb_palette[3 * j + 1];
            expand_palette_input[i]->b_palette[j] = expand_palette_input[i]->rgb_palette[3 * j + 2];
        }
        random_init_2D<png_byte>(rows, columns * 4, expand_palette_output[i]->output_buf);
    }

    config = (config_t *)expand_palette_config;
    input = (input_t **)expand_palette_input;
    output = (output_t **)expand_palette_output;

    return count;
}
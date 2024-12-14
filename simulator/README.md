# In-Cache Computing Simulator for MVE ISA Extension

This directory provides **an in-cache computing simulator** on top of Ramulator \[1\] for data-parallel applications based on the Multi-Dimensional Vector ISA Extension *(MVE)* ISA extension.
This simulator is trace-driven.
To generate traces, you can use [our DynamoRIO client](/tools/DynamoRIO/samples/inscount.cpp) on Arm-based processors.

## Trace Format

Traces provided to this simulator contain 4 types of instructions:

1. **Scalar CPU Load and Store Instructions:**
Each line of CPU load and store trace must be in this format:

  - `<load/store> <address> <num-cpuinst>`

      - where `<num-cpuinst>` shows the number of compute instructions before this trace line.


2. **MVE Config Operations:** The length of vector instructions is #SA x 256.
*MVE* changes the vector dimension and length using the following instructions:

  - `<opcode> -1 -1 -1 <config-dim> <config-val> <num-cpuinst>`


3. **MVE Vector Memory Operations:** Vector memory operations include multiple strides with the following configuration.
If the memory operation is random, base random addresses follow the trace line.

  - `<opcode> <dst> <address> -1 <stride3> <stride2> <stride1> <stride0> <num-cpuinst> [<base-addr-0> <base-addr-1> ...]`


4. **MVE Vector Compute Operations:**

  - `<opcode> <dst> <src1> <src2> -1 -1 -1 -1 <num-cpuinst>`

For a list of available opcodes, refer to [data directory](/data).

## Usage

Compile the simulator using these commands:

    $ cd ramulator
    $ bash make_all.sh

The bash script generate various executables for different ISAs (RVV or MVE) and the following in-cache computing schemes:

  - Bit-Serial (bs)
  - Bit-Hybrid (bh)
  - Bit-Parallel (bp)
  - Associative (ac)

Run the simulator using this command:

    $ ./ramulator_<ISA>_<SCHEME> <config-file> --mode=MVE --core=1 <core-type> --stats <stat-file> <trace-file>

where:
  - `<config-file>` contains DRAM config, proportional CPU and DRAM frequency, and in-cache computing level.
  You can find [an example in the config directory](/configs/LPDDR4-config-MVE.cfg) for in-L2 computing.
  
  - `<core-type>` Currently, we support 3 core configurations: `prime`, `gold`, `silver` which are Cortex-A76 cores of Qualcomm Snapdragon 855 SoC.

  - `<stat-file>` is where your simulation results will be stored.

  - `<trace-file>` contains the trace of dynamic instructions.


## Debug

To activate debugging logs, please compile ramulator with `DEBUG` flag:
  
    $ bash make_all_verbose.sh

## Functionality

Refer to [this readme](MVE_README.md) for more information about the MVE implementation.

## References

\[1\] Y. Kim, W. Yang and O. Mutlu, "Ramulator: A Fast and Extensible DRAM Simulator," in *IEEE Computer Architecture Letters*, 2016.

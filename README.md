# MVE Simulator for Mobile In-Cache Computing

This repo provides **an in-cache computing simulator** on top of Ramulator \[1\] for data-parallel applications based on the Multi-Dimensional Vector ISA Extension *(MVE)* ISA extension.

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

## Citation

If you use *MVE*, please cite this paper:

> Alireza Khadem, Daichi Fujiki, Hilbert Chen, Yufeng Gu, Nishil Talati, Scott Mahlke, and Reetuparna Das.
> *Multi-Dimensional Vector ISA Extension for Mobile In-Cache Computing*,
> In 2025 IEEE International Symposium on High-Performance Computer Architecture (HPCA)

```
@inproceedings{mve,
  title={Multi-Dimensional Vector ISA Extension for Mobile In-Cache Computing},
  author={Khadem, Alireza and Fujiki, Daichi and Chen, Hilbert and Gu, Yufeng and Talati, Nishil and Mahlke, Scott and Das, Reetuparna},
  booktitle={2025 IEEE International Symposium on High-Performance Computer Architecture (HPCA)}, 
  year={2025}
}
```

## Issues and bug reporting

We appreciate any feedback and suggestions from the community.
Feel free to raise an issue or submit a pull request on Github.
For assistance in using Swan, please contact: Alireza Khadem (arkhadem@umich.edu)

## Licensing

*MVE* simulator is available under a [MIT license](/LICENSE).
For Ramulator, please refer to their [GitHub repo](https://github.com/CMU-SAFARI/ramulator).

## Acknowledgement

This work was supported in part by the NSF under CAREER-1652294 and NSF-1908601 awards.

## References

[\[1\] Kim et al. *Ramulator: A Fast and Extensible DRAM Simulator.* IEEE CAL
2015.](https://people.inf.ethz.ch/omutlu/pub/ramulator_dram_simulator-ieee-cal15.pdf)


## Processor (Core)

Processor calls `Cache::send`.

## Cache

1. **`Cache::send`**
    - `Cache::MVE_incoming_req_queue`: instructions sent by processor, outstanding to the cache
    1. Cache Checks `Cache::MVE_incoming_req_queue` not full
    2. Cache adds instruction to `Cache::MVE_incoming_req_queue`, taking current cycle + latency of that cache level as the receive time.

2. **`Cache::tick`**
    1. Every cycle, cache goes through list of outstanding instructions.
    2. If receiving new instructions is not *`Cache::receive_locked`* because of prior random memory accesses or dictionary operations, and
    2. If any current cycle is >= of intruction's receive time:
        - `Cache::MVE_controller` is called for the instruction.
        - Instruction is removed from `Cache::MVE_incoming_req_queue`.

3. **`Cache::MVE_controller`**
    - decodes the instruction.
    1. If instruction is **config**, it is decoded and finished in this function.
    2. If instruction is **random memory access or dictionary**, it requires additional data loaded from the memory (random base addresses or dictionary values). Thus, Cache calls `Cache::random_dict_access_decoder`:
        - This function calculates required data to be loaded for random memory accesses or dictionaries. It pushes the instruction to `Cache::MVE_random_dict_to_mem_ops`.
        - If possible, these instructions are sent to the same level of cache. If not, they are stored in `Cache::self_retry_list` for later.
        - Cache is *`Cache::receive_locked`* until the required data is received.
        - When `Cache::callback` is called for these data accesses, Cache checks if `Cache::MVE_random_dict_to_mem_ops` is empty. It unlocks *`Cache::receive_locked`* and calls `Cache::instrinsic_decoder` for this instruction.
    3. If instruction is **compute** or **strided memory access** or a random memory access / dictionary which data is received, Cache calls `Cache::instrinsic_decoder`:
        - If instruction is strided memory access, start and end addresses are calculated.
        - For all instructions, number of control blocks that will receive a TINY intrinsic is calculated and stored in `Cache::MVE_vop_to_num_sop`. If this number is zero (because of masking), instruction is immediately called back.
        - For each TINY intrinsic that targets a control block, `Cache::intrinsic_computer` is called.
    4. `Cache::intrinsic_computer` calculated latency and energy for this TINY intrinsic. It stores the TINY intrinsic along with its latency to each control block queue, i.e., `Cache::MVE_compute_queue`.

4. **`Cache::tick` - Step 1 (Start Computing Instruction)**
    - If the instruction at the head of the `MVE_compute_queue` has not been started computing yet, i.e., `Cache::last_MVE_instruction_computed` of the control block is *false*, it **can* be started computing:
        - If the instruction is a **dictionary**, Cache checks if the `Cache::crossbar_locked` is *false*.
    - Control blocks starts computing the instruction by keeping track of current cycle in `Cache::last_MVE_instruction_compute_clk` and setting `Cache::last_MVE_instruction_computed` to *true*. In addition, if the instruction is a dictionary, `Cache::crossbar_locked` is set to *true*;

5. **`Cache::tick` - Step 2 (Finish Computing Instruction - Unpack Memory Access)**
    - Cache checks whether the instruction at the head of the `MVE_compute_queue` is finished computing if:
        1. It has been started computing, i.e., `Cache::last_MVE_instruction_computed` of the control block is *true*.
        2. and if it is a memory access, it is not *unpacked* yet, i.e., `Cache::last_MVE_instruction_sent` of the control block is *false*.
    - If it's a **memory access**:
        - It is *unpacked* to memory operations. Individual memory operations for a MVE instruction are kept in `Cache::MVE_op_to_mem_ops`.
        - After unpacking, cache goes through memory operations. If possible, it sends them to the same level of the cache.
        - Cache sets `Cache::last_MVE_instruction_sent` to *true*, showing that the instruction is unpacked.
    - If it's a **dictionary**, `Cache::crossbar_locked` is set to *False*.
    - If it's a **dictionary** or **compute** instruction, computation is finished by setting `Cache::last_MVE_instruction_computed` to *false*. Here, control block calls `Cache::callbacker` for the computed TINY intrinsic.
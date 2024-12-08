[
    [ "Annotations", "page_annotations.html", [
      [ "Annotation Types", "page_annotations.html#autotoc_md18", null ],
      [ "Annotation Samples from Debug Builds", "page_annotations.html#autotoc_md19", [
        [ "Unix x64 Expression", "page_annotations.html#autotoc_md20", null ],
        [ "Unix x64 Statement", "page_annotations.html#autotoc_md21", null ],
        [ "Unix x86 Expression", "page_annotations.html#autotoc_md22", null ],
        [ "Unix x86 Statement", "page_annotations.html#autotoc_md23", null ],
        [ "Windows x64 Expression", "page_annotations.html#autotoc_md24", null ],
        [ "Windows x64 Statement", "page_annotations.html#autotoc_md25", null ],
        [ "Windows x86 Expression", "page_annotations.html#autotoc_md26", null ],
        [ "Windows x86 Statement", "page_annotations.html#autotoc_md27", null ]
      ] ],
      [ "Samples of Special Cases", "page_annotations.html#autotoc_md28", [
        [ "Nested annotations: Windows x64 (/Ox /GL)", "page_annotations.html#autotoc_md29", null ],
        [ "Nested annotations with shared label: Windows x64 (/Ox /GL)", "page_annotations.html#autotoc_md30", null ]
      ] ],
      [ "Detection Algorithms", "page_annotations.html#autotoc_md31", [
        [ "Unix", "page_annotations.html#autotoc_md32", null ],
        [ "Windows x86", "page_annotations.html#autotoc_md33", null ],
        [ "Windows x64", "page_annotations.html#autotoc_md34", null ]
      ] ],
      [ "Instrumentation Algorithms", "page_annotations.html#autotoc_md35", [
        [ "Expression", "page_annotations.html#autotoc_md36", null ],
        [ "Statement", "page_annotations.html#autotoc_md37", null ]
      ] ]
    ] ],
    [ "ARM Port", "page_arm_port.html", [
      [ "Decoder/Encoder Approach", "page_arm_port.html#autotoc_md38", null ],
      [ "DynamoRIO IR for ARM: IR decisions", "page_arm_port.html#autotoc_md39", null ],
      [ "Code refactoring: names", "page_arm_port.html#autotoc_md40", null ],
      [ "Code refactoring: opcodes", "page_arm_port.html#autotoc_md41", [
        [ "Sharing OPSZ_ constants", "page_arm_port.html#autotoc_md42", null ],
        [ "ARM vs x86 Arch macro", "page_arm_port.html#autotoc_md43", null ]
      ] ],
      [ "TLS Access", "page_arm_port.html#autotoc_md44", null ],
      [ "ASM Approach", "page_arm_port.html#autotoc_md45", null ],
      [ "Register enum", "page_arm_port.html#autotoc_md46", null ],
      [ "TLS via Stolen Register: Interactions with tools", "page_arm_port.html#autotoc_md47", [
        [ "Proposal 1:", "page_arm_port.html#autotoc_md48", null ],
        [ "Proposal 2: do not allow meta instructions to use stolen reg except as TLS base.", "page_arm_port.html#autotoc_md49", null ],
        [ "Proposal 3: fully expose stolen reg and have API routine to access the stolen value.  Burden is on tool to not mess up stolen reg.", "page_arm_port.html#autotoc_md50", null ],
        [ "Proposal 4: mangle app stolen reg before showing to tool", "page_arm_port.html#autotoc_md51", null ],
        [ "Proposal 5: isolate all use of stolen reg to single-instr bb and swap stolen reg for that bb", "page_arm_port.html#autotoc_md52", null ],
        [ "Metrics:", "page_arm_port.html#autotoc_md53", null ],
        [ "Proposal 6: swap stolen reg around each app/tool insr that uses it; TLS reg is virtual", "page_arm_port.html#autotoc_md54", null ],
        [ "Discussion:", "page_arm_port.html#autotoc_md55", null ],
        [ "Suggestion:", "page_arm_port.html#autotoc_md56", null ],
        [ "OS/TLS/Steal reg", "page_arm_port.html#autotoc_md57", null ],
        [ "Mangle App TLS", "page_arm_port.html#autotoc_md58", null ]
      ] ],
      [ "Direct Link Reachability", "page_arm_port.html#autotoc_md59", null ],
      [ "IT Block Handling", "page_arm_port.html#autotoc_md60", null ],
      [ "Handle the app switching between ARM and Thumb", "page_arm_port.html#autotoc_md61", null ],
      [ "IT Blocks Part 2: Splitting", "page_arm_port.html#autotoc_md62", [
        [ "Further discussion 4/23/15", "page_arm_port.html#autotoc_md63", null ]
      ] ],
      [ "Conditional Syscall", "page_arm_port.html#autotoc_md64", null ]
    ] ],
    [ "AArch64 Port", "page_aarch64_port.html", [
      [ "Introduction to AArch64", "page_aarch64_port.html#autotoc_md12", null ],
      [ "IR decisions", "page_aarch64_port.html#autotoc_md13", null ],
      [ "Encoder/decoder", "page_aarch64_port.html#autotoc_md14", null ],
      [ "Stolen register", "page_aarch64_port.html#autotoc_md15", null ],
      [ "Reachability", "page_aarch64_port.html#autotoc_md16", null ],
      [ "Self-modifying code", "page_aarch64_port.html#autotoc_md17", null ]
    ] ],
    [ "Linking Far Fragments on AArch64", "page_aarch64_far.html", [
      [ "Background", "page_aarch64_far.html#autotoc_md0", [
        [ "Existing Implementation", "page_aarch64_far.html#autotoc_md1", [
          [ "ARM32", "page_aarch64_far.html#autotoc_md2", null ],
          [ "AArch64", "page_aarch64_far.html#autotoc_md3", null ],
          [ "x86", "page_aarch64_far.html#autotoc_md4", null ]
        ] ]
      ] ],
      [ "Problem Statement", "page_aarch64_far.html#autotoc_md5", null ],
      [ "Proposed Solutions", "page_aarch64_far.html#autotoc_md6", [
        [ "Option 1: Append load-and-branch-to-target instrs to existing exit stub", "page_aarch64_far.html#autotoc_md7", null ],
        [ "Option 2: Reuse code between fcache_return/linked stub", "page_aarch64_far.html#autotoc_md8", null ],
        [ "Option 3: Landing Pads", "page_aarch64_far.html#autotoc_md9", null ],
        [ "Option 4: Reuse data slot between fcache_return/linked stub", "page_aarch64_far.html#autotoc_md10", null ]
      ] ],
      [ "Conclusion", "page_aarch64_far.html#autotoc_md11", null ]
    ] ],
    [ "JIT Optimization", "page_jitopt.html", [
      [ "Branch Content", "page_jitopt.html#autotoc_md184", null ],
      [ "Commit Workflow", "page_jitopt.html#autotoc_md185", null ],
      [ "Code Quality", "page_jitopt.html#autotoc_md186", null ]
    ] ],
    [ "Restartable Sequences", "page_rseq.html", [
      [ "Background", "page_rseq.html#autotoc_md227", [
        [ "Why must DynamoRIO handle Restartable Sequences specially?", "page_rseq.html#autotoc_md228", null ],
        [ "RSEQ API/ABI", "page_rseq.html#autotoc_md229", null ]
      ] ],
      [ "Challenges for Handling RSEQ in DR", "page_rseq.html#autotoc_md230", [
        [ "Identifying Restartable Sequences", "page_rseq.html#autotoc_md231", null ],
        [ "Running Restartable Sequences under DynamoRIO", "page_rseq.html#autotoc_md232", null ]
      ] ],
      [ "The \"Run Twice\" Solution", "page_rseq.html#autotoc_md233", [
        [ "Running Twice", "page_rseq.html#autotoc_md234", null ],
        [ "Rejected Alternatives", "page_rseq.html#autotoc_md235", [
          [ "Emulate Restartable Sequences using CPU Affinity", "page_rseq.html#autotoc_md236", null ],
          [ "Emulate Restartable Sequences using a Per-Sequence Mutex", "page_rseq.html#autotoc_md237", null ],
          [ "Have DR Generate Restartable Sequence Fragment Groups", "page_rseq.html#autotoc_md238", null ]
        ] ]
      ] ],
      [ "\"Run Twice\" Implementation Details", "page_rseq.html#autotoc_md239", [
        [ "Fallback: Disable Rseq", "page_rseq.html#autotoc_md240", null ],
        [ "Identifying Rseq Sequences", "page_rseq.html#autotoc_md241", null ],
        [ "First (Instrumented) Execution", "page_rseq.html#autotoc_md242", null ],
        [ "Second (Restartable) Execution", "page_rseq.html#autotoc_md243", [
          [ "Application State Barrier", "page_rseq.html#autotoc_md244", null ],
          [ "Target the Start, Not the Abort Handler", "page_rseq.html#autotoc_md245", null ],
          [ "Local Copy", "page_rseq.html#autotoc_md246", null ],
          [ "Marking Restartable", "page_rseq.html#autotoc_md247", null ],
          [ "Where to Locate the rseq_cs?", "page_rseq.html#autotoc_md248", null ],
          [ "Clearing the Rseq Bounds", "page_rseq.html#autotoc_md249", null ],
          [ "Abort Handler", "page_rseq.html#autotoc_md250", null ],
          [ "Obtaining Cache Addresses for rseq_cs", "page_rseq.html#autotoc_md251", null ],
          [ "Testing", "page_rseq.html#autotoc_md252", null ]
        ] ]
      ] ],
      [ "Future Work", "page_rseq.html#autotoc_md253", null ],
      [ "Limitations", "page_rseq.html#autotoc_md254", null ],
      [ "Citations", "page_rseq.html#autotoc_md255", null ]
    ] ],
    [ "Exclusive Monitors", "page_ldstex.html", [
      [ "Overview", "page_ldstex.html#autotoc_md187", null ],
      [ "What is an Exclusive Monitor?", "page_ldstex.html#autotoc_md188", null ],
      [ "The Problem with Instrumenting Exclusive Monitors", "page_ldstex.html#autotoc_md189", [
        [ "Consequences: Infinite Loop!", "page_ldstex.html#autotoc_md190", null ],
        [ "Problem Not Limited by Tool Type", "page_ldstex.html#autotoc_md191", null ],
        [ "Problem Exacerbated by Intervening Branches", "page_ldstex.html#autotoc_md192", null ]
      ] ],
      [ "Initial Implemented Solution: Just Avoid Clean Calls", "page_ldstex.html#autotoc_md193", null ],
      [ "Proposed Solution A: Super-Instruction", "page_ldstex.html#autotoc_md194", null ],
      [ "Proposed Solution B: Compare-and-Swap Simulation", "page_ldstex.html#autotoc_md195", null ],
      [ "Proposed Solution C: Atomic Add Conversion", "page_ldstex.html#autotoc_md196", null ],
      [ "Proposed Solution D: Run Twice", "page_ldstex.html#autotoc_md197", null ],
      [ "Combining Solutions", "page_ldstex.html#autotoc_md198", null ],
      [ "Decision: Compare-and-Swap", "page_ldstex.html#autotoc_md199", null ],
      [ "Issue Tracker References", "page_ldstex.html#autotoc_md200", null ]
    ] ],
    [ "Using an External Decoder", "page_external_decoder.html", [
      [ "Motivation", "page_external_decoder.html#autotoc_md165", null ],
      [ "Implementation", "page_external_decoder.html#autotoc_md166", null ],
      [ "Requirements", "page_external_decoder.html#autotoc_md167", null ],
      [ "Potential Decoders", "page_external_decoder.html#autotoc_md168", [
        [ "XED", "page_external_decoder.html#autotoc_md169", null ],
        [ "LLVM", "page_external_decoder.html#autotoc_md170", null ]
      ] ],
      [ "Concerns", "page_external_decoder.html#autotoc_md171", null ]
    ] ],
    [ "Emulating x86 Scatter and Gather Instructions", "page_scatter_gather_emulation.html", [
      [ "Background", "page_scatter_gather_emulation.html#autotoc_md256", null ],
      [ "Problem Statement", "page_scatter_gather_emulation.html#autotoc_md257", null ],
      [ "Design", "page_scatter_gather_emulation.html#autotoc_md258", [
        [ "Scatter/gather Instruction Expansion", "page_scatter_gather_emulation.html#autotoc_md259", null ],
        [ "Drreg Support For Multi-phase Reservations", "page_scatter_gather_emulation.html#autotoc_md260", [
          [ "State Restoration For Drreg", "page_scatter_gather_emulation.html#autotoc_md261", null ]
        ] ],
        [ "Simplifying Instrumentation For Emulated Instructions", "page_scatter_gather_emulation.html#autotoc_md262", null ],
        [ "Support For Vector Reservation", "page_scatter_gather_emulation.html#autotoc_md263", null ],
        [ "Using The Expansion In DR Clients", "page_scatter_gather_emulation.html#autotoc_md264", null ]
      ] ],
      [ "Testing On Large Apps", "page_scatter_gather_emulation.html#autotoc_md265", null ]
    ] ]
],
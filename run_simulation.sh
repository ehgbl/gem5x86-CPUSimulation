#!/bin/bash
# Gem5 x86 Simulation Runner Script
# =================================

echo "Gem5 x86 CPU Architecture Simulation"
echo "===================================="

# Check if gem5 is available
if [ -z "$GEM5_ROOT" ]; then
    echo "Error: GEM5_ROOT environment variable not set."
    echo "Please set it to your gem5 installation directory:"
    echo "  export GEM5_ROOT=/path/to/gem5"
    exit 1
fi

# Check if Python script exists
if [ ! -f "gem5_x86_simulation.py" ]; then
    echo "Error: gem5_x86_simulation.py not found in current directory."
    exit 1
fi

# Create results directory
mkdir -p results

# Build workload if not exists
if [ ! -f "sample_workload" ] || [ ! -f "hello" ]; then
    echo "Building workload executables..."
    chmod +x build_workload.sh
    ./build_workload.sh
fi

# Default simulation parameters
CPU_TYPE="timing"
CPU_CORES=2
CPU_FREQ="2GHz"
L1I_SIZE="32kB"
L1D_SIZE="32kB"
L2_SIZE="256kB"
MEM_SIZE="4GB"
MEM_LATENCY=100
MAX_INSTS=1000000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-type)
            CPU_TYPE="$2"
            shift 2
            ;;
        --cpu-cores)
            CPU_CORES="$2"
            shift 2
            ;;
        --cpu-freq)
            CPU_FREQ="$2"
            shift 2
            ;;
        --l1i-size)
            L1I_SIZE="$2"
            shift 2
            ;;
        --l1d-size)
            L1D_SIZE="$2"
            shift 2
            ;;
        --l2-size)
            L2_SIZE="$2"
            shift 2
            ;;
        --mem-size)
            MEM_SIZE="$2"
            shift 2
            ;;
        --mem-latency)
            MEM_LATENCY="$2"
            shift 2
            ;;
        --max-insts)
            MAX_INSTS="$2"
            shift 2
            ;;
        --workload)
            WORKLOAD="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cpu-type TYPE      CPU type (atomic, timing, o3) [default: timing]"
            echo "  --cpu-cores N        Number of CPU cores [default: 2]"
            echo "  --cpu-freq FREQ      CPU frequency [default: 2GHz]"
            echo "  --l1i-size SIZE      L1 instruction cache size [default: 32kB]"
            echo "  --l1d-size SIZE      L1 data cache size [default: 32kB]"
            echo "  --l2-size SIZE       L2 cache size [default: 256kB]"
            echo "  --mem-size SIZE      Memory size [default: 4GB]"
            echo "  --mem-latency N      Memory latency in cycles [default: 100]"
            echo "  --max-insts N        Maximum instructions [default: 1000000]"
            echo "  --workload BINARY    Workload binary [default: sample_workload]"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set default workload if not specified
if [ -z "$WORKLOAD" ]; then
    WORKLOAD="sample_workload"
fi

# Check if workload exists
if [ ! -f "$WORKLOAD" ]; then
    echo "Error: Workload '$WORKLOAD' not found."
    echo "Available workloads:"
    ls -la | grep -E "^-.*[^.]$" | awk '{print "  " $9}'
    exit 1
fi

echo "Simulation Configuration:"
echo "  CPU Type: $CPU_TYPE"
echo "  CPU Cores: $CPU_CORES"
echo "  CPU Frequency: $CPU_FREQ"
echo "  L1I Cache: $L1I_SIZE"
echo "  L1D Cache: $L1D_SIZE"
echo "  L2 Cache: $L2_SIZE"
echo "  Memory Size: $MEM_SIZE"
echo "  Memory Latency: $MEM_LATENCY cycles"
echo "  Max Instructions: $MAX_INSTS"
echo "  Workload: $WORKLOAD"
echo ""

# Run the simulation
echo "Starting simulation..."
python3 gem5_x86_simulation.py \
    --cpu-type "$CPU_TYPE" \
    --cpu-cores "$CPU_CORES" \
    --cpu-freq "$CPU_FREQ" \
    --l1i-size "$L1I_SIZE" \
    --l1d-size "$L1D_SIZE" \
    --l2-size "$L2_SIZE" \
    --mem-size "$MEM_SIZE" \
    --mem-latency "$MEM_LATENCY" \
    --max-insts "$MAX_INSTS" \
    --binary "$WORKLOAD" \
    --output-dir "results"

# Check if simulation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Simulation completed successfully!"
    echo "Results saved to: results/"
    
    # Run performance analysis if available
    if [ -f "performance_analysis.py" ]; then
        echo ""
        echo "Running performance analysis..."
        python3 performance_analysis.py --stats-file "stats.txt" --output-dir "results"
    fi
    
    echo ""
    echo "Available result files:"
    ls -la results/
else
    echo "Simulation failed!"
    exit 1
fi

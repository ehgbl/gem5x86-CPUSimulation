


import os
import sys
import argparse
import time
from pathlib import Path


GEM5_ROOT = os.environ.get('GEM5_ROOT', '/opt/gem5')
sys.path.append(os.path.join(GEM5_ROOT, 'src', 'python'))
sys.path.append(os.path.join(GEM5_ROOT, 'src', 'python', 'm5'))

try:
    import m5
    from m5.objects import *
    from m5.util import addToPath
    from m5.stats import periodicStatDump
except ImportError:
    print("Error: gem5 not found. Please set GEM5_ROOT environment variable.")
    print("Example: export GEM5_ROOT=/path/to/gem5")
    sys.exit(1)


addToPath(os.path.join(GEM5_ROOT, 'configs'))

try:
    from common import Options
    from common import Simulation
    from common import MemConfig
    from common import CpuConfig
    from common import ObjectList
    from common.Caches import *
    from common.CpuConfig import *
except ImportError:
    print("Error: gem5 configs not found. Please check GEM5_ROOT path.")
    sys.exit(1)


class Gem5X86Simulation:
    """Main simulation class for x86 CPU architecture analysis."""
    
    def __init__(self, args):
        self.args = args
        self.system = None
        self.cpu = None
        self.memory = None
        self.cache_hierarchy = None
        
    def create_system(self):
        """Create the complete x86 system with CPU, memory, and bus."""
        print("Creating x86 system architecture...")
        
        self.system = System()
        self.system.clk_domain = SrcClockDomain()
        self.system.clk_domain.clock = self.args.cpu_freq
        self.system.clk_domain.voltage_domain = VoltageDomain()
        self.system.mem_mode = 'timing'
        self.system.mem_ranges = [AddrRange(self.args.mem_size)]
        
        self.create_cpu()
        self.create_memory_subsystem()
        self.create_cache_hierarchy()
        self.connect_system()
        
        print("System architecture created successfully!")
        
    def create_cpu(self):
        """Create and configure the x86 CPU."""
        print(f"Creating x86 CPU with {self.args.cpu_cores} cores...")
        

        if self.args.cpu_type == 'timing':
            self.cpu = [TimingSimpleCPU(cpu_id=i) for i in range(self.args.cpu_cores)]
        elif self.args.cpu_type == 'o3':
            self.cpu = [DerivO3CPU(cpu_id=i) for i in range(self.args.cpu_cores)]
        else:
            self.cpu = [AtomicSimpleCPU(cpu_id=i) for i in range(self.args.cpu_cores)]
        

        for i, cpu in enumerate(self.cpu):
            cpu.clk_domain = SrcClockDomain()
            cpu.clk_domain.clock = self.args.cpu_freq
            cpu.clk_domain.voltage_domain = VoltageDomain()
            

            if hasattr(cpu, 'max_insts_any_thread'):
                cpu.max_insts_any_thread = self.args.max_insts
            if hasattr(cpu, 'max_insts_all_threads'):
                cpu.max_insts_all_threads = self.args.max_insts
                
        print(f"CPU configuration: {self.args.cpu_type} with {self.args.cpu_cores} cores")
        
    def create_memory_subsystem(self):
        """Create DRAM memory subsystem with memory controller."""
        print("Creating DRAM memory subsystem...")
        

        self.system.mem_ctrl = MemCtrl()
        self.system.mem_ctrl.dram = DDR3_1600_8x8()
        

        self.system.mem_ctrl.dram.range = AddrRange(self.args.mem_size)
        self.system.mem_ctrl.port = self.system.membus.mem_side_ports
        

        self.system.mem_ctrl.dram.tRCD = self.args.mem_latency
        self.system.mem_ctrl.dram.tCL = self.args.mem_latency
        self.system.mem_ctrl.dram.tRP = self.args.mem_latency
        
        print(f"Memory configuration: {self.args.mem_size // (1024**3)}GB DDR3-1600")
        print(f"Memory latency: {self.args.mem_latency} cycles")
        
    def create_cache_hierarchy(self):
        """Create multi-level cache hierarchy."""
        print("Creating cache hierarchy...")
        

        self.system.cpu_icache = L1ICache(size=self.args.l1i_size, 
                                        assoc=self.args.l1i_assoc)
        

        self.system.cpu_dcache = L1DCache(size=self.args.l1d_size,
                                        assoc=self.args.l1d_assoc)
        

        self.system.l2cache = L2Cache(size=self.args.l2_size,
                                    assoc=self.args.l2_assoc)
        

        if self.args.l3_size > 0:
            self.system.l3cache = L3Cache(size=self.args.l3_size,
                                        assoc=self.args.l3_assoc)
        
        print(f"Cache hierarchy: L1I={self.args.l1i_size}KB, L1D={self.args.l1d_size}KB, "
              f"L2={self.args.l2_size}KB" + (f", L3={self.args.l3_size}KB" if self.args.l3_size > 0 else ""))
        
    def connect_system(self):
        """Connect all system components via system bus."""
        print("Connecting system components...")
        

        self.system.membus = SystemXBar()
        self.system.membus.clk_domain = SrcClockDomain()
        self.system.membus.clk_domain.clock = self.args.bus_freq
        self.system.membus.clk_domain.voltage_domain = VoltageDomain()
        

        for i, cpu in enumerate(self.cpu):

            cpu.icache_port = self.system.cpu_icache.cpu_side
            cpu.dcache_port = self.system.cpu_dcache.cpu_side
            

            self.system.cpu_icache.mem_side = self.system.l2cache.cpu_side
            self.system.cpu_dcache.mem_side = self.system.l2cache.cpu_side
            

        if self.args.l3_size > 0:
            self.system.l2cache.mem_side = self.system.l3cache.cpu_side
            self.system.l3cache.mem_side = self.system.membus.cpu_side_ports
        else:
            self.system.l2cache.mem_side = self.system.membus.cpu_side_ports
        

        self.system.mem_ctrl.port = self.system.membus.mem_side_ports
        
        print("System bus configuration complete!")
        
    def create_workload(self):
        """Create and configure workload for simulation."""
        print("Setting up workload...")
        

        process = Process()
        process.cmd = [self.args.binary]
        if self.args.args:
            process.cmd.extend(self.args.args.split())
        

        for i, cpu in enumerate(self.cpu):
            cpu.workload = process
            cpu.createThreads()
            
        print(f"Workload configured: {self.args.binary}")
        
    def run_simulation(self):
        """Run the complete simulation."""
        print("Starting simulation...")
        print("=" * 50)
        

        m5.stats.addStatVisitor(m5.stats.SimPointStatVisitor())
        

        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        

        if self.args.checkpoint_restore:
            print("Restoring from checkpoint...")
            m5.instantiate(self.args.checkpoint_restore)
        else:
            print("Instantiating system...")
            m5.instantiate()
            

        print(f"Running simulation for {self.args.max_insts} instructions...")
        exit_event = m5.simulate()
        
        print("Simulation completed!")
        print(f"Exit event: {exit_event.getCause()}")
        
        return exit_event
        
    def analyze_performance(self):
        """Analyze and display performance metrics."""
        print("\n" + "=" * 50)
        print("PERFORMANCE ANALYSIS")
        print("=" * 50)
        

        self.analyze_cache_performance()
        

        self.analyze_memory_performance()
        

        self.analyze_cpu_performance()
        

        self.analyze_system_performance()
        
    def analyze_cache_performance(self):
        """Analyze cache hit rates and performance."""
        print("\n--- CACHE PERFORMANCE ---")
        
        try:

            l1i_hits = m5.stats.get('system.cpu_icache.hits')
            l1i_misses = m5.stats.get('system.cpu_icache.misses')
            l1i_hit_rate = l1i_hits / (l1i_hits + l1i_misses) * 100 if (l1i_hits + l1i_misses) > 0 else 0
            
            print(f"L1 Instruction Cache:")
            print(f"  Hits: {l1i_hits}")
            print(f"  Misses: {l1i_misses}")
            print(f"  Hit Rate: {l1i_hit_rate:.2f}%")
            

            l1d_hits = m5.stats.get('system.cpu_dcache.hits')
            l1d_misses = m5.stats.get('system.cpu_dcache.misses')
            l1d_hit_rate = l1d_hits / (l1d_hits + l1d_misses) * 100 if (l1d_hits + l1d_misses) > 0 else 0
            
            print(f"L1 Data Cache:")
            print(f"  Hits: {l1d_hits}")
            print(f"  Misses: {l1d_misses}")
            print(f"  Hit Rate: {l1d_hit_rate:.2f}%")
            

            l2_hits = m5.stats.get('system.l2cache.hits')
            l2_misses = m5.stats.get('system.l2cache.misses')
            l2_hit_rate = l2_hits / (l2_hits + l2_misses) * 100 if (l2_hits + l2_misses) > 0 else 0
            
            print(f"L2 Cache:")
            print(f"  Hits: {l2_hits}")
            print(f"  Misses: {l2_misses}")
            print(f"  Hit Rate: {l2_hit_rate:.2f}%")
            
        except Exception as e:
            print(f"Error analyzing cache performance: {e}")
            
    def analyze_memory_performance(self):
        """Analyze memory bandwidth and latency."""
        print("\n--- MEMORY PERFORMANCE ---")
        
        try:

            mem_reads = m5.stats.get('system.mem_ctrl.num_reads')
            mem_writes = m5.stats.get('system.mem_ctrl.num_writes')
            mem_read_bytes = m5.stats.get('system.mem_ctrl.bytes_read')
            mem_write_bytes = m5.stats.get('system.mem_ctrl.bytes_written')
            
            print(f"Memory Operations:")
            print(f"  Read Requests: {mem_reads}")
            print(f"  Write Requests: {mem_writes}")
            print(f"  Data Read: {mem_read_bytes / (1024**2):.2f} MB")
            print(f"  Data Written: {mem_write_bytes / (1024**2):.2f} MB")
            

            avg_mem_latency = m5.stats.get('system.mem_ctrl.avg_memory_latency')
            print(f"  Average Memory Latency: {avg_mem_latency:.2f} cycles")
            
        except Exception as e:
            print(f"Error analyzing memory performance: {e}")
            
    def analyze_cpu_performance(self):
        """Analyze CPU instruction throughput and efficiency."""
        print("\n--- CPU PERFORMANCE ---")
        
        try:
            for i, cpu in enumerate(self.cpu):
                insts = m5.stats.get(f'system.cpu{i}.numInsts')
                cycles = m5.stats.get(f'system.cpu{i}.numCycles')
                cpi = cycles / insts if insts > 0 else 0
                ipc = insts / cycles if cycles > 0 else 0
                
                print(f"CPU {i}:")
                print(f"  Instructions: {insts}")
                print(f"  Cycles: {cycles}")
                print(f"  CPI (Cycles per Instruction): {cpi:.2f}")
                print(f"  IPC (Instructions per Cycle): {ipc:.2f}")
                
        except Exception as e:
            print(f"Error analyzing CPU performance: {e}")
            
    def analyze_system_performance(self):
        """Analyze overall system performance metrics."""
        print("\n--- SYSTEM PERFORMANCE ---")
        
        try:

            sim_ticks = m5.curTick()
            sim_time = sim_ticks / 1e12  # Convert to seconds
            
            print(f"Simulation Time: {sim_time:.2f} seconds")
            print(f"Simulation Ticks: {sim_ticks}")
            

            mem_util = m5.stats.get('system.mem_ctrl.bw_util')
            print(f"Memory Bandwidth Utilization: {mem_util:.2f}%")
            
        except Exception as e:
            print(f"Error analyzing system performance: {e}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Gem5 x86 CPU Architecture Simulation')
    

    parser.add_argument('--cpu-type', default='timing', 
                       choices=['atomic', 'timing', 'o3'],
                       help='CPU type (default: timing)')
    parser.add_argument('--cpu-cores', type=int, default=1,
                       help='Number of CPU cores (default: 1)')
    parser.add_argument('--cpu-freq', default='2GHz',
                       help='CPU frequency (default: 2GHz)')
    parser.add_argument('--max-insts', type=int, default=1000000,
                       help='Maximum instructions to simulate (default: 1000000)')
    

    parser.add_argument('--l1i-size', default='32kB',
                       help='L1 instruction cache size (default: 32kB)')
    parser.add_argument('--l1d-size', default='32kB',
                       help='L1 data cache size (default: 32kB)')
    parser.add_argument('--l1i-assoc', type=int, default=2,
                       help='L1 instruction cache associativity (default: 2)')
    parser.add_argument('--l1d-assoc', type=int, default=2,
                       help='L1 data cache associativity (default: 2)')
    parser.add_argument('--l2-size', default='256kB',
                       help='L2 cache size (default: 256kB)')
    parser.add_argument('--l2-assoc', type=int, default=8,
                       help='L2 cache associativity (default: 8)')
    parser.add_argument('--l3-size', default='0kB',
                       help='L3 cache size (default: 0kB)')
    parser.add_argument('--l3-assoc', type=int, default=16,
                       help='L3 cache associativity (default: 16)')
    

    parser.add_argument('--mem-size', default='2GB',
                       help='Memory size (default: 2GB)')
    parser.add_argument('--mem-latency', type=int, default=100,
                       help='Memory latency in cycles (default: 100)')
    

    parser.add_argument('--bus-freq', default='1GHz',
                       help='System bus frequency (default: 1GHz)')
    

    parser.add_argument('--binary', default='hello',
                       help='Binary to execute (default: hello)')
    parser.add_argument('--args', default='',
                       help='Arguments for the binary')
    

    parser.add_argument('--checkpoint-restore', default=None,
                       help='Checkpoint to restore from')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for results')
    
    return parser.parse_args()


def main():
    """Main simulation function."""
    print("Gem5 x86 CPU Architecture Simulation")
    print("====================================")
    

    args = parse_arguments()
    

    os.makedirs(args.output_dir, exist_ok=True)
    

    sim = Gem5X86Simulation(args)
    
    try:

        sim.create_system()
        

        sim.create_workload()
        

        exit_event = sim.run_simulation()
        

        sim.analyze_performance()
        
        print("\nSimulation completed successfully!")
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == '__main__':
    sys.exit(main())


"""
Configuration Examples for Gem5 x86 Simulation
==============================================

This file contains pre-configured simulation setups for different
scenarios and use cases, demonstrating various computer architecture
concepts and performance trade-offs.
"""

import subprocess
import sys
from pathlib import Path

class SimulationConfigs:
    """Pre-configured simulation setups for different scenarios."""
    
    @staticmethod
    def run_config(config_name, **kwargs):
        """Run a simulation with the specified configuration."""
        cmd = [sys.executable, "gem5_x86_simulation.py"]
        
        for key, value in kwargs.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        print(f"Running configuration: {config_name}")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 50)
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running {config_name}: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    @staticmethod
    def high_performance_server():
        """High-performance server configuration."""
        return {
            'cpu_type': 'o3',
            'cpu_cores': 8,
            'cpu_freq': '4GHz',
            'l1i_size': '128kB',
            'l1d_size': '128kB',
            'l1i_assoc': 4,
            'l1d_assoc': 4,
            'l2_size': '2MB',
            'l2_assoc': 16,
            'l3_size': '16MB',
            'l3_assoc': 32,
            'mem_size': '32GB',
            'mem_latency': 50,
            'max_insts': 10000000,
            'binary': 'sample_workload'
        }
    
    @staticmethod
    def mobile_processor():
        """Mobile/low-power processor configuration."""
        return {
            'cpu_type': 'timing',
            'cpu_cores': 4,
            'cpu_freq': '1.5GHz',
            'l1i_size': '32kB',
            'l1d_size': '32kB',
            'l1i_assoc': 2,
            'l1d_assoc': 2,
            'l2_size': '256kB',
            'l2_assoc': 8,
            'l3_size': '2MB',
            'l3_assoc': 16,
            'mem_size': '4GB',
            'mem_latency': 150,
            'max_insts': 5000000,
            'binary': 'sample_workload'
        }
    
    @staticmethod
    def embedded_system():
        """Embedded system configuration."""
        return {
            'cpu_type': 'atomic',
            'cpu_cores': 1,
            'cpu_freq': '500MHz',
            'l1i_size': '16kB',
            'l1d_size': '16kB',
            'l1i_assoc': 2,
            'l1d_assoc': 2,
            'l2_size': '128kB',
            'l2_assoc': 4,
            'l3_size': '0kB',
            'mem_size': '512MB',
            'mem_latency': 200,
            'max_insts': 1000000,
            'binary': 'hello'
        }
    
    @staticmethod
    def cache_sensitivity_study():
        """Configuration for cache sensitivity analysis."""
        return {
            'cpu_type': 'timing',
            'cpu_cores': 1,
            'cpu_freq': '2GHz',
            'l1i_size': '64kB',
            'l1d_size': '64kB',
            'l1i_assoc': 4,
            'l1d_assoc': 4,
            'l2_size': '1MB',
            'l2_assoc': 16,
            'l3_size': '8MB',
            'l3_assoc': 32,
            'mem_size': '8GB',
            'mem_latency': 100,
            'max_insts': 5000000,
            'binary': 'sample_workload'
        }
    
    @staticmethod
    def memory_bandwidth_test():
        """Configuration optimized for memory bandwidth testing."""
        return {
            'cpu_type': 'o3',
            'cpu_cores': 4,
            'cpu_freq': '3GHz',
            'l1i_size': '32kB',
            'l1d_size': '32kB',
            'l1i_assoc': 2,
            'l1d_assoc': 2,
            'l2_size': '256kB',
            'l2_assoc': 8,
            'l3_size': '0kB',
            'mem_size': '16GB',
            'mem_latency': 75,
            'max_insts': 8000000,
            'binary': 'sample_workload'
        }
    
    @staticmethod
    def multi_core_scalability():
        """Configuration for multi-core scalability analysis."""
        return {
            'cpu_type': 'timing',
            'cpu_cores': 16,
            'cpu_freq': '2.5GHz',
            'l1i_size': '64kB',
            'l1d_size': '64kB',
            'l1i_assoc': 4,
            'l1d_assoc': 4,
            'l2_size': '512kB',
            'l2_assoc': 16,
            'l3_size': '32MB',
            'l3_assoc': 64,
            'mem_size': '64GB',
            'mem_latency': 100,
            'max_insts': 20000000,
            'binary': 'sample_workload'
        }


def run_all_configurations():
    """Run all pre-configured simulations."""
    configs = {
        'high_performance_server': SimulationConfigs.high_performance_server,
        'mobile_processor': SimulationConfigs.mobile_processor,
        'embedded_system': SimulationConfigs.embedded_system,
        'cache_sensitivity_study': SimulationConfigs.cache_sensitivity_study,
        'memory_bandwidth_test': SimulationConfigs.memory_bandwidth_test,
        'multi_core_scalability': SimulationConfigs.multi_core_scalability
    }
    
    results = {}
    
    for name, config_func in configs.items():
        print(f"\n{'='*60}")
        print(f"Running configuration: {name.upper()}")
        print(f"{'='*60}")
        
        config = config_func()
        success = SimulationConfigs.run_config(name, **config)
        results[name] = success
        
        if success:
            print(f"✓ {name} completed successfully")
        else:
            print(f"✗ {name} failed")
    

    print(f"\n{'='*60}")
    print("SIMULATION SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name:30} {status}")
    
    print(f"\nCompleted: {successful}/{total} simulations")
    
    return results


def run_configuration_series():
    """Run a series of related configurations for comparison."""
    

    print("Running cache size sensitivity study...")
    cache_sizes = ['16kB', '32kB', '64kB', '128kB', '256kB']
    
    for size in cache_sizes:
        config = {
            'cpu_type': 'timing',
            'cpu_cores': 1,
            'cpu_freq': '2GHz',
            'l1i_size': size,
            'l1d_size': size,
            'l2_size': '512kB',
            'mem_size': '4GB',
            'max_insts': 2000000,
            'binary': 'sample_workload'
        }
        
        SimulationConfigs.run_config(f"cache_size_{size}", **config)
    

    print("\nRunning CPU frequency scaling study...")
    frequencies = ['1GHz', '1.5GHz', '2GHz', '2.5GHz', '3GHz']
    
    for freq in frequencies:
        config = {
            'cpu_type': 'timing',
            'cpu_cores': 1,
            'cpu_freq': freq,
            'l1i_size': '32kB',
            'l1d_size': '32kB',
            'l2_size': '256kB',
            'mem_size': '4GB',
            'max_insts': 2000000,
            'binary': 'sample_workload'
        }
        
        SimulationConfigs.run_config(f"freq_{freq}", **config)
    

    print("\nRunning memory latency sensitivity study...")
    latencies = [50, 75, 100, 150, 200, 300]
    
    for latency in latencies:
        config = {
            'cpu_type': 'timing',
            'cpu_cores': 1,
            'cpu_freq': '2GHz',
            'l1i_size': '32kB',
            'l1d_size': '32kB',
            'l2_size': '256kB',
            'mem_size': '4GB',
            'mem_latency': latency,
            'max_insts': 2000000,
            'binary': 'sample_workload'
        }
        
        SimulationConfigs.run_config(f"latency_{latency}", **config)


def main():
    """Main function for running configuration examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gem5 Configuration Examples')
    parser.add_argument('--config', choices=[
        'high_performance_server', 'mobile_processor', 'embedded_system',
        'cache_sensitivity_study', 'memory_bandwidth_test', 'multi_core_scalability'
    ], help='Run specific configuration')
    parser.add_argument('--all', action='store_true', help='Run all configurations')
    parser.add_argument('--series', action='store_true', help='Run configuration series')
    
    args = parser.parse_args()
    
    if args.config:
        config_func = getattr(SimulationConfigs, args.config)
        config = config_func()
        SimulationConfigs.run_config(args.config, **config)
    elif args.all:
        run_all_configurations()
    elif args.series:
        run_configuration_series()
    else:
        print("Available configurations:")
        print("  high_performance_server  - High-performance server setup")
        print("  mobile_processor         - Mobile/low-power processor")
        print("  embedded_system          - Embedded system configuration")
        print("  cache_sensitivity_study  - Cache sensitivity analysis")
        print("  memory_bandwidth_test    - Memory bandwidth testing")
        print("  multi_core_scalability   - Multi-core scalability")
        print("\nUse --all to run all configurations or --series for sensitivity studies")


if __name__ == '__main__':
    main()

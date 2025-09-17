



echo "Building sample workload for gem5 x86 simulation..."


if ! command -v gcc &> /dev/null; then
    echo "Error: gcc compiler not found. Please install gcc."
    exit 1
fi


echo "Compiling sample_workload.c..."
gcc -O2 -o sample_workload sample_workload.c -lm

if [ $? -eq 0 ]; then
    echo "Successfully compiled sample_workload"
    echo "Executable: ./sample_workload"
else
    echo "Error: Failed to compile sample_workload.c"
    exit 1
fi


echo "Creating hello world program..."
cat > hello.c << 'EOF'


int main() {
    printf("Hello from gem5 x86 simulation!\n");
    printf("This is a simple workload for testing CPU performance.\n");
    
    // Simple computation
    int sum = 0;
    for (int i = 0; i < 1000000; i++) {
        sum += i;
    }
    
    printf("Computation result: %d\n", sum);
    return 0;
}
EOF

gcc -O2 -o hello hello.c

if [ $? -eq 0 ]; then
    echo "Successfully compiled hello world program"
    echo "Executable: ./hello"
else
    echo "Error: Failed to compile hello.c"
    exit 1
fi

echo ""
echo "Build complete! Available executables:"
echo "  - ./sample_workload (comprehensive workload)"
echo "  - ./hello (simple workload)"
echo ""
echo "You can now run the gem5 simulation with:"
echo "  python gem5_x86_simulation.py --binary sample_workload"
echo "  python gem5_x86_simulation.py --binary hello"

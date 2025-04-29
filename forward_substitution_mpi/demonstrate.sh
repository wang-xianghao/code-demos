# Build 
make clean
make 

# Show the speedup of pipeline
echo mpirun -np 1 ./solve 10
mpirun -np 1 ./solve 10
echo
echo mpirun -np 2 ./solve 10
mpirun -np 2 ./solve 10
echo
echo mpirun -np 3 ./solve 10
mpirun -np 3 ./solve 10
echo
echo mpirun -np 4 ./solve 10
mpirun -np 4 ./solve 10
echo
echo mpirun -np 5 ./solve 10
mpirun -np 5 ./solve 10
echo
echo mpirun -np 6 ./solve 10
mpirun -np 6 ./solve 10
echo
echo mpirun -np 7 ./solve 10
mpirun -np 7 ./solve 10
echo
echo mpirun -np 8 ./solve 10
mpirun -np 8 ./solve 10
echo
echo mpirun -np 9 ./solve 10
mpirun -np 9 ./solve 10
echo
echo mpirun -np 10 ./solve 1
mpirun -np 10 ./solve 10
echo
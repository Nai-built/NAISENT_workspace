cd NeurologicalLibrary/bridge; cmake -S . -B build -A x64; cmake --build build --config Release -j; cd ../..
cd OptimizedNeurologicalLibrary; cmake -S . -B build -A x64; cmake --build build --config Release -j; cd ..
cd CudaNeurologicalLibrary; cmake -S . -B build -A x64; cmake --build build --config Release -j; cd ..
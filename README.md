this project is made with:
 - DotNet WinForms (C#)
 - Pybind11 (Python <-> C++23)
 - CMake (C++23)
 - CUDA (C++17)

Powershell commands to build the 3 libraries:
cd NeurologicalLibrary/bridge; cmake -S . -B build -A x64; cmake --build build --config Release -j; cd ../..
cd OptimizedNeurologicalLibrary; cmake -S . -B build -A x64; cmake --build build --config Release -j; cd ..
cd CudaNeurologicalLibrary; cmake -S . -B build -A x64; cmake --build build --config Release -j; cd ..

Run showcases:
py SHOWCASES/BASIC_SHAPE_RECOGNITION_CPU.py
py SHOWCASES/BETA_NAISENT_BALL_SEEKER_CPU.py
py SHOWCASES/LSTM_MATH_TEST_CPU.py
py SHOWCASES/NAISENT_ELM_CPU.py
py SHOWCASES/NAISENT_LM_CUDA.py
py SHOWCASES/NAISENT_SLM_CUDA.py
py SHOWCASES/SHAPE_RECOGNITION_CPU.py

Make sure that your terminal's path is set exactly to NAISENT_workspace

The core idea of this project was to learn and understand Machine Learning by building it from scratch
So I've built 3 different libraries in 3 seperate stages:
 - NeurologicalLibrary (NL)
    . The absolute beginning for me
    . I've learned in it how Dense Layers work and how to chain them to make Deep Neural Networks
    . How Convolutional Layers and pools work
    . How Recursive Layers (specifically LSTMs) work
    . And also Activation Functions
    . I've also tipped toes into Graph Layers but couldn't run a successful experiment, so I removed it
    . This library was the first time I made an image recognintion model, and also one that can play a simple bounce ball game
    . Was also the first time I made an optimizer like Adam for training
    . Save/load system for the model .json files

 - OptimizedNeurologicalLibrary (ONL)
    . Here things started to get a bit more serious
    . I've gotten way deeper into how C++ works and how we can optimize its performance
    . I've made faster Dense Layers
    . Faster Convolutional Layers
    . And faster LSTMs
    . Merged Activation Functions into the layers' own activation/gradient functions
    . After that, I got into Transformers (similar concept to Graph Layers, but this time it was successful!)
    . I optimized the training loop for image recognition
    . I made a simple experimental language model that can that it's "NAISENT" with the Transformer system I've made

 - CudaNeurologicalLibrary (CNL)
    . My most precious one so far
    . For the first time, I've got into Cuda kernels!
    . I've learned how Cuda interacts with data through the CPU, Memory and GPU
    . I've learned how to optimize it using shared memory
    . For this one, I went right ahead to build a language model system
    . First, I made Dense Layer Cuda kernels
    . Then I went into Norm Layers (RMS)
    . SCC (Sine/Cosine Cycle) positional embedding kernels
    . Multi-head Masked Self Attention kernels (split into multiple optimized Cuda files)
    . The ability to place sub chains to assemble the transformer architecture properly
    . Adam optimizer in Cuda Kernels
    . And obviously, Activation Functions (Cuda kernels)
    . First time adding the Residual mechanic as a visible variable in the Python side
    . Almost all of these were made in ONL already, but it wasn't with Cuda to use the GPU and it was juggled up together awkwardly. I'm much more proud of this one
    . Was when I made a proper tokenizer system in Python

The libraries are made in C++
and they're used by the Python side via Pybind11
I made the shape recognition and bounce ball environments in C# with WinForms
CUDA to use the GPU in the library CNL
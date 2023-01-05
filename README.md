# CUDA Game of life

-----
This repository implements straightforward Game of life algorithm.

Main purpose of this repository is to explore CUDA OpenGL interoperability. Usage of CUDA alongside with 
OpenGL provides opportunity to perform computation and render result within same device. This approach 
helps to minimize data transfers overheads.

## Building application

----
Application could be build with `cmake` using Visual Studio generators:

```bash
cd GameOfLife_CUDA_GL
cmake -S . -B <build_directory>
cmake --build <build_directory> --target gol_cuda_gl --config <config_name>
```

`build_directory` - name of output directory for build <br>
`config_name` - build config name (e.g. `Debug`, `Release`)

All necessary dependencies will be linked or build automatically alongside with the application.

## Dependencies

----
* CUDA capable GPU
* CUDA runtime
* GLFW
* GLAD
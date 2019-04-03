#include "cuda.h"
#include <stdio.h>

int main(){
    //Inicjalizajca drivera - za nim uruchomimy jaka kolwiek funkcje z Driver API
    cuInit(0);
    
    //Pobranie handlera do devica
    //(moze byc kilka urzadzen. Tutaj zakladamy, ze jest conajmniej jedno)
    CUdevice cuDevice;
    CUresult res = cuDeviceGet(&cuDevice, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot acquire device 0\n"); 
        exit(1);
    }

    //Tworzy kontext
    CUcontext cuContext;
    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if (res != CUDA_SUCCESS){
        printf("cannot create context\n");
        exit(1);
    }

    //Tworzy modul z pliku binarnego "gcd.ptx"
    CUmodule cuModule = (CUmodule)0;
    res = cuModuleLoad(&cuModule, "helloWorldDriverAPI.ptx");
    if (res != CUDA_SUCCESS) {
        printf("cannot load module: %d\n", res);  
        exit(1); 
    }

    //Pobiera handler kernela z modulu
    CUfunction helloWorld;
    res = cuModuleGetFunction(&helloWorld, cuModule, "HelloWorld");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }

    int blocks_per_grid = 4;
    int threads_per_block = 5;

    void* args[] = {};
    res = cuLaunchKernel(helloWorld, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
    }

    cuCtxDestroy(cuContext);

    return 0;
}

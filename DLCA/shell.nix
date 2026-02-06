{ pkgs ? import <nixpkgs> { config = { allowUnfree = true; }; } }:

let
  cuda = pkgs.cudaPackages_12_8;

  cuptiPath = "${cuda.cudatoolkit}/extras/CUPTI/lib64";

  baseLibPath = pkgs.lib.makeLibraryPath [
    cuda.cudatoolkit
    pkgs.stdenv.cc.cc
    "/run/opengl-driver"
  ];

  pyEnv = pkgs.python312.withPackages (ps: with ps; [
    # core jupyter bits
    jupyterlab
    notebook
    ipykernel

    # your libs
    matplotlib
#    jax
    jax-cuda12-plugin
    jaxlib-bin
    pillow
    numpy
    optax
    einops
    flax
  ]);
in
pkgs.mkShell {
  buildInputs = [
    pyEnv

    # CUDA 12.8 runtime bits
    cuda.cudatoolkit
    cuda.cudnn
    cuda.nccl
  ];

  LD_LIBRARY_PATH = "${cuptiPath}:${baseLibPath}";
  CUDA_PATH = cuda.cudatoolkit;
  CUDA_HOME = cuda.cudatoolkit;
}


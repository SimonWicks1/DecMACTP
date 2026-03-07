# Installation

Note:
<b>If you neeed to re-install GNARL, follow the following steps:
`cd GNARL -> pip install . -> cd root path ->pip install -e BenchMARL`
If you use `pip install .` in Benchmarl, it would make all your code in the conda lib, so you can not change the code to run. (Like Gnarl, you have to first pip install to make the change synchronize to the conda lib)
</b>
## [MAPPO Onpolicy](https://github.com/marlbenchmark/on-policy)
python: Try 3.10
Conda: marl
Too much errors. There is no tutorials for the repo.
Pause, if benchamrl cannot work, back to rewrite mappo based on this repo

---
**V2:Multi-Train with ARC"**
1. Keep the version of dynamic callback for IPPO and IGNARL 
   1. MAPPO, IPPO and UGNARL adopt shared trunk: IPPO ang IGNARL use the same shared layer, share by critic and actor, but in update critic detach; MAPPPO use duplicate feature extractor and encode; actor and critic update their own networks. 
2. Change the Train.py: Add parser to pass the random seed
3. Read the default.yaml to get environment config. All random seeds are trained in the same dataset seed.
---
**V1: first commit** First commit: The baselines code version in the supervision discussion 06/03/2026.
We want to make the GIPPO and GMAPPO algorithm adopt share featureExtractor and Encode. 
This need leads to V2.

## [BenchMARL](https://github.com/facebookresearch/BenchMARL?tab=readme-ov-file#install)
First install GNARLy
Then install benchmarl
there would be such error
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gnarl 0.1 requires torch==2.6.0, but you have torch 2.9.1 which is incompatible.
```
but if you check the libs with the command
```
conda list
```
There would be
```
# packages in environment at /home/pemb7543/anaconda3/envs/marl:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
absl-py                   2.3.1                    pypi_0    pypi
aiofiles                  25.1.0                   pypi_0    pypi
aiohappyeyeballs          2.6.1                    pypi_0    pypi
aiohttp                   3.13.3                   pypi_0    pypi
aiosignal                 1.4.0                    pypi_0    pypi
annotated-types           0.7.0                    pypi_0    pypi
antlr4-python3-runtime    4.9.3                    pypi_0    pypi
array-record              0.8.3                    pypi_0    pypi
astunparse                1.6.3                    pypi_0    pypi
attrs                     25.4.0                   pypi_0    pypi
av                        13.1.0                   pypi_0    pypi
benchmarl                 1.5.1                    pypi_0    pypi
binutils                  2.44                 h4852527_5    conda-forge
binutils_impl_linux-64    2.44                 h4b9a079_2  
binutils_linux-64         2.44                 hc03a8fd_2  
boost-cpp                 1.85.0               h3c6214e_4    conda-forge
bzip2                     1.0.8                h5eee18b_6  
c-ares                    1.34.6               hb03c661_0    conda-forge
c-compiler                1.11.0               h4d9bdce_0    conda-forge
ca-certificates           2026.1.4             hbd8a1cb_0    conda-forge
certifi                   2026.1.4                 pypi_0    pypi
charset-normalizer        3.4.4                    pypi_0    pypi
chex                      0.1.91                   pypi_0    pypi
click                     8.3.1                    pypi_0    pypi
cloudpickle               3.1.2                    pypi_0    pypi
cmake                     4.2.1                hc85cc9f_0    conda-forge
compilers                 1.11.0               ha770c72_0    conda-forge
conda-gcc-specs           14.3.0              he8ccf15_16    conda-forge
contourpy                 1.3.3                    pypi_0    pypi
curl                      8.18.0               h4e3cde8_0    conda-forge
cxx-compiler              1.11.0               hfcd1e18_0    conda-forge
cycler                    0.12.1                   pypi_0    pypi
dm-clrs                   2.0.3                    pypi_0    pypi
dm-haiku                  0.0.16                   pypi_0    pypi
dm-tree                   0.1.9                    pypi_0    pypi
docstring-parser          0.17.0                   pypi_0    pypi
einops                    0.8.1                    pypi_0    pypi
etils                     1.13.0                   pypi_0    pypi
expat                     2.7.3                h3385a95_0  
farama-notifications      0.0.4                    pypi_0    pypi
filelock                  3.20.2                   pypi_0    pypi
flatbuffers               25.12.19                 pypi_0    pypi
flax                      0.12.2                   pypi_0    pypi
fonttools                 4.61.1                   pypi_0    pypi
fortran-compiler          1.11.0               h9bea470_0    conda-forge
frozenlist                1.8.0                    pypi_0    pypi
fsspec                    2026.1.0                 pypi_0    pypi
gast                      0.7.0                    pypi_0    pypi
gcc                       14.3.0               h76bdaa0_7    conda-forge
gcc_impl_linux-64         14.3.0               hd9e9e21_7    conda-forge
gcc_linux-64              14.3.0              h298d278_17    conda-forge
gfortran                  14.3.0               he448592_7    conda-forge
gfortran_impl_linux-64    14.3.0              h1a219da_16    conda-forge
gfortran_linux-64         14.3.0              h9ce9316_17    conda-forge
gitdb                     4.0.12                   pypi_0    pypi
gitpython                 3.1.46                   pypi_0    pypi
gnarl                     0.1                      pypi_0    pypi
google-pasta              0.2.0                    pypi_0    pypi
googleapis-common-protos  1.72.0                   pypi_0    pypi
grpcio                    1.76.0                   pypi_0    pypi
gxx                       14.3.0               he448592_7    conda-forge
gxx_impl_linux-64         14.3.0               he663afc_7    conda-forge
gxx_linux-64              14.3.0              h310e576_17    conda-forge
gymnasium                 1.2.3                    pypi_0    pypi
h5py                      3.15.1                   pypi_0    pypi
humanize                  4.15.0                   pypi_0    pypi
hydra-core                1.3.2                    pypi_0    pypi
icu                       75.1                 he02047a_0    conda-forge
idna                      3.11                     pypi_0    pypi
immutabledict             4.2.2                    pypi_0    pypi
importlib-metadata        8.7.1                    pypi_0    pypi
importlib-resources       6.5.2                    pypi_0    pypi
jax                       0.8.2                    pypi_0    pypi
jaxlib                    0.8.2                    pypi_0    pypi
jinja2                    3.1.6                    pypi_0    pypi
jmp                       0.0.4                    pypi_0    pypi
keras                     3.13.0                   pypi_0    pypi
kernel-headers_linux-64   6.12.0               he073ed8_5    conda-forge
keyutils                  1.6.3                hb9d3cd8_0    conda-forge
kiwisolver                1.4.9                    pypi_0    pypi
krb5                      1.21.3               h659f571_0    conda-forge
ld_impl_linux-64          2.44                 h153f514_2  
libboost                  1.85.0               h0ccab89_4    conda-forge
libboost-devel            1.85.0               h00ab1b0_4    conda-forge
libboost-headers          1.85.0               ha770c72_4    conda-forge
libclang                  18.1.1                   pypi_0    pypi
libcurl                   8.18.0               h4e3cde8_0    conda-forge
libedit                   3.1.20250104    pl5321h7949ede_0    conda-forge
libev                     4.33                 hd590300_2    conda-forge
libexpat                  2.7.3                hecca717_0    conda-forge
libffi                    3.4.4                h6a678d5_1  
libgcc                    15.2.0               h69a1729_7  
libgcc-devel_linux-64     14.3.0             h85bb3a7_107    conda-forge
libgcc-ng                 15.2.0               h166f726_7  
libgfortran5              15.2.0              h68bc16d_16    conda-forge
libgomp                   15.2.0               h4751f2c_7  
liblzma                   5.8.1                hb9d3cd8_2    conda-forge
liblzma-devel             5.8.1                hb9d3cd8_2    conda-forge
libnghttp2                1.67.0               had1ee68_0    conda-forge
libnsl                    2.0.0                h5eee18b_0  
libsanitizer              14.3.0               hd08acf3_7    conda-forge
libssh2                   1.11.1               hcf80075_0    conda-forge
libstdcxx                 15.2.0               h39759b7_7  
libstdcxx-devel_linux-64  14.3.0             h85bb3a7_107    conda-forge
libstdcxx-ng              15.2.0               hc03a8fd_7  
libuuid                   1.41.5               h5eee18b_0  
libuv                     1.51.0               hb03c661_1    conda-forge
libxcb                    1.17.0               h9b100fa_0  
libzlib                   1.3.1                hb25bd0a_0  
make                      4.4.1                hb9d3cd8_2    conda-forge
markdown                  3.10                     pypi_0    pypi
markdown-it-py            4.0.0                    pypi_0    pypi
markupsafe                3.0.3                    pypi_0    pypi
matplotlib                3.10.8                   pypi_0    pypi
mdurl                     0.1.2                    pypi_0    pypi
ml-collections            1.1.0                    pypi_0    pypi
ml-dtypes                 0.5.4                    pypi_0    pypi
mpmath                    1.3.0                    pypi_0    pypi
msgpack                   1.1.2                    pypi_0    pypi
multidict                 6.7.0                    pypi_0    pypi
namex                     0.1.0                    pypi_0    pypi
ncurses                   6.5                  h2d0b736_3    conda-forge
nest-asyncio              1.6.0                    pypi_0    pypi
networkx                  3.6.1                    pypi_0    pypi
numpy                     2.1.3                    pypi_0    pypi
nvidia-cublas-cu12        12.8.4.1                 pypi_0    pypi
nvidia-cuda-cupti-cu12    12.8.90                  pypi_0    pypi
nvidia-cuda-nvrtc-cu12    12.8.93                  pypi_0    pypi
nvidia-cuda-runtime-cu12  12.8.90                  pypi_0    pypi
nvidia-cudnn-cu12         9.10.2.21                pypi_0    pypi
nvidia-cufft-cu12         11.3.3.83                pypi_0    pypi
nvidia-cufile-cu12        1.13.1.3                 pypi_0    pypi
nvidia-curand-cu12        10.3.9.90                pypi_0    pypi
nvidia-cusolver-cu12      11.7.3.90                pypi_0    pypi
nvidia-cusparse-cu12      12.5.8.93                pypi_0    pypi
nvidia-cusparselt-cu12    0.7.1                    pypi_0    pypi
nvidia-nccl-cu12          2.27.5                   pypi_0    pypi
nvidia-nvjitlink-cu12     12.8.93                  pypi_0    pypi
nvidia-nvshmem-cu12       3.3.20                   pypi_0    pypi
nvidia-nvtx-cu12          12.8.90                  pypi_0    pypi
omegaconf                 2.3.0                    pypi_0    pypi
openssl                   3.6.0                h26f9b46_0    conda-forge
opt-einsum                3.4.0                    pypi_0    pypi
optax                     0.2.6                    pypi_0    pypi
optree                    0.18.0                   pypi_0    pypi
orbax-checkpoint          0.11.31                  pypi_0    pypi
orjson                    3.11.5                   pypi_0    pypi
packaging                 25.0                     pypi_0    pypi
pandas                    2.3.3                    pypi_0    pypi
patch                     2.8               hb03c661_1002    conda-forge
pillow                    12.1.0                   pypi_0    pypi
pip                       25.3               pyhc872135_0  
platformdirs              4.5.1                    pypi_0    pypi
promise                   2.3                      pypi_0    pypi
propcache                 0.4.1                    pypi_0    pypi
protobuf                  5.29.5                   pypi_0    pypi
psutil                    7.2.1                    pypi_0    pypi
pthread-stubs             0.3                  h0ce48e5_1  
pulp                      3.3.0                    pypi_0    pypi
pyarrow                   22.0.0                   pypi_0    pypi
pydantic                  2.12.5                   pypi_0    pypi
pydantic-core             2.41.5                   pypi_0    pypi
pygments                  2.19.2                   pypi_0    pypi
pyparsing                 3.3.1                    pypi_0    pypi
python                    3.11.14              h6fa692b_0  
python-dateutil           2.9.0.post0              pypi_0    pypi
python_abi                3.11                    3_cp311  
pytz                      2025.2                   pypi_0    pypi
pyvers                    0.1.0                    pypi_0    pypi
pyyaml                    6.0.3                    pypi_0    pypi
readline                  8.3                  hc2a1206_0  
requests                  2.32.5                   pypi_0    pypi
rhash                     1.4.6                hb9d3cd8_1    conda-forge
rich                      14.2.0                   pypi_0    pypi
sb3-contrib               2.7.1                    pypi_0    pypi
scipy                     1.16.3                   pypi_0    pypi
sentry-sdk                2.49.0                   pypi_0    pypi
setuptools                80.9.0          py311h06a4308_0  
simple-parsing            0.1.7                    pypi_0    pypi
simplejson                3.20.2                   pypi_0    pypi
six                       1.17.0                   pypi_0    pypi
smmap                     5.0.2                    pypi_0    pypi
sqlite                    3.51.1               he0a8d7e_0  
stable-baselines3         2.7.1                    pypi_0    pypi
sympy                     1.14.0                   pypi_0    pypi
sysroot_linux-64          2.39                 hc4b9eeb_5    conda-forge
tabulate                  0.9.0                    pypi_0    pypi
tensorboard               2.19.0                   pypi_0    pypi
tensorboard-data-server   0.7.2                    pypi_0    pypi
tensordict                0.10.0                   pypi_0    pypi
tensorflow                2.19.0                   pypi_0    pypi
tensorflow-io-gcs-filesystem 0.37.1                   pypi_0    pypi
tensorflow-metadata       1.17.0                   pypi_0    pypi
tensorstore               0.1.80                   pypi_0    pypi
termcolor                 3.3.0                    pypi_0    pypi
tfds-nightly              4.9.8.dev202504060044          pypi_0    pypi
tk                        8.6.15               h54e0aa7_0  
toml                      0.10.2                   pypi_0    pypi
toolz                     1.1.0                    pypi_0    pypi
torch                     2.9.1                    pypi_0    pypi
torch-geometric           2.6.1                    pypi_0    pypi
torchrl                   0.10.1                   pypi_0    pypi
torchvision               0.24.1                   pypi_0    pypi
tqdm                      4.67.1                   pypi_0    pypi
treescope                 0.1.10                   pypi_0    pypi
triton                    3.5.1                    pypi_0    pypi
typing-extensions         4.15.0                   pypi_0    pypi
typing-inspection         0.4.2                    pypi_0    pypi
tzdata                    2025.3                   pypi_0    pypi
urllib3                   2.6.3                    pypi_0    pypi
wandb                     0.23.1                   pypi_0    pypi
werkzeug                  3.1.5                    pypi_0    pypi
wheel                     0.45.1          py311h06a4308_0  
wrapt                     2.0.1                    pypi_0    pypi
xorg-libx11               1.8.12               h9b100fa_1  
xorg-libxau               1.0.12               h9b100fa_0  
xorg-libxdmcp             1.1.5                h9b100fa_0  
xorg-xorgproto            2024.1               h5eee18b_1  
xz                        5.8.1                hbcc6ac9_2    conda-forge
xz-gpl-tools              5.8.1                hbcc6ac9_2    conda-forge
xz-tools                  5.8.1                hb9d3cd8_2    conda-forge
yarl                      1.22.0                   pypi_0    pypi
zipp                      3.23.0                   pypi_0    pypi
zlib                      1.3.1                hb25bd0a_0  
zstd                      1.5.7                hb78ec9c_6    conda-forge
```
Then, you run `python /home/pemb7543/DeC_MACTP/GNARL-MACTP/scripts/test_mactp_env.py`, it works


- What we discussed last meeting
  Previously, we constructed the MACTP environment and extended the GNARL algorithm to the MACTP problem. However, we found that as the number of agents grows, the performance of the GNARL algorithm decreases, which may be because the joint action space increases exponentially with the number of agents. But before we move to the decentralised MARL setting, we try to find a suitable mechanism for this MACTP environment (i.e., find a suitable reward function for the agents). The details of the 2 reward mechnisim:
  Reward Shaping
    Case 1:
    * Remove the self-loop for all non-goal nodes, and keep the stay cost to be 0 for goal nodes
    * Add the Penalty for failure(-15*num_goals_unvisited)
    Case 2:
    * Add agent termination 
    * Remove all self-loops
  As it is a long break, we also planned to explore the decentralised MARL setting (try MAPPO+GNN in the environment).
- What’s your current state
  I have wrote the environments with the two reward mechnisim we discussed in last meeting, and updated it on Github repo. The training with multiple random seeds is waiting for the connection of GOALS server because I have the issue of connecting ori-waln (which means I cannnot connect with GOALS-server), but the single-seed training on my laptop has shown some promising results. I am working on explore the decentralised MARL setting now.
- Where you want to by the next time we meet
  I wish I could complete the decentralised MARL setting before next meeting.
- And what you’re likely to want to discuss when we next meet
  1. Choice of reward shaping
  2. how to set the evaluation metrics
  3. Challengs with decentralised MARL setting

# Construct MACTP in BenchMARL

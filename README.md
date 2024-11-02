# GAMES101-Windows

GAMES101 作業的 windows 環境，使用 msvc 與 CMake 進行建置

# How to build?

```
git clone https://github.com/Mes0903/GAMES101-Windows.git
cd GAMES101-Windows
git submodule init
git submodule update
mkdir build && cd build
cmake ..
cmake --build . --config Release --target ALL_BUILD -j 8
```

目前 Debug 的情況尚有未知原因導致無法建置
## Сборка с типом double 
cmake -B build -DUSE_FLOAT=OFF
cmake --build build
./build/exercise01

## Сборка с типом float
cmake -B build -DUSE_FLOAT=ON
cmake --build build
./build/exercise01

## Вывод
float дал вывод: 0.291951
double дал вывод: 4.80487e-11

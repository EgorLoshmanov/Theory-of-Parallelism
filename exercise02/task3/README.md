## Создать папку сборки
mkdir build

cd build

## Сгенерировать проект 

### С оптимизацией 
cmake -DCMAKE_BUILD_TYPE=Release ..

### Без оптимизации 
cmake -DCMAKE_BUILD_TYPE=Debug ..

## Собрать
cmake --build . -j

## Запуск
./simple_iteration

## Задание числа потоков
export OMP_NUM_THREADS=<число потоков>
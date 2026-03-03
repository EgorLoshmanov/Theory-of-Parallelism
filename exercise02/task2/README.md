## Создать папку сборки
mkdir build

cd build

## Сгенерировать проект 
cmake -DCMAKE_BUILD_TYPE=Release ..

## Собрать
cmake --build . -j

## Запуск
./integration

## Задание числа потоков
export OMP_NUM_THREADS=<число потоков>
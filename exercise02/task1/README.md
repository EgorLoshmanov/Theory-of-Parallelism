## Создать папку сборки
mkdir build

cd build

## Сгенерировать проект
cmake -DCMAKE_BUILD_TYPE=Release ..

## Собрать
cmake --build . -j

## Запуск c параметрами по умолчанию (m = 20000, n = 20000)
./dgemv

## Запуск с указанием размеров:
./dgemv 40000 40000

## Задание числа потоков
export OMP_NUM_THREADS=<число потоков>
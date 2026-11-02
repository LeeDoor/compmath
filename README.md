# Лабораторная работа по линейному программированию

## Запуск

```bash
./run.sh
```

При первом запуске `run.sh` создаёт виртуальное окружение и устанавливает зависимости из `requirements.txt`. Затем появится числовое меню; выберите задачу, введя её номер. Выйти можно пунктом `0`.

## Структура

- `problem.py` — создание и проверка задачи `A*x = b`, целевая функция `c*x` и направление `max`/`min`.
- `display_matrix.py`, `display_augmented_matrix.py` — вывод матриц.
- `display_function.py`, `display_max.py`, `display_min.py` — формирование и вывод целевых функций.
- `simplex.py`, `dual_simplex.py`, `artificial_basis.py`, `transport.py`, `primal_dual.py`, `optimality.py` — отдельный файл для каждого алгоритма.
- `canonical_form.py`, `standard_form.py` — отдельные точки входа для приведения задачи к канонической и стандартной формам.

Четыре метода решения СЛАУ с целевой функцией принимают `A`, `b`, коэффициенты `c` и направление `sense` (`"max"` или `"min"`), а после реализации должны возвращать вектор решения (`numpy.matrix`). Методы преобразования форм и алгоритмы намеренно оставлены незаполненными, чтобы их можно было реализовать по правилам вашего курса. Матрицы задачи хранятся как `numpy.matrix`, а для точного отображения используются объекты SymPy.

## Подготовка задачи

```python
from linear_programming.problem import create_problem
from linear_programming.simplex import solve_simplex

problem = create_problem(A=[[1, 1], [2, 1]], b=[4, 5], c=[3, 2], sense="max")

# После реализации метода:
# solution = solve_simplex(problem.A, problem.b, problem.c, problem.sense)
```

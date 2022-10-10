import streamlit as st

st.header('Функции потерь, активации и число нейронов выходного слоя')

st.markdown(
    '''
|| Задача | Функция потерь | Функция в 🔥PyTorch | Функция Активации | Функция в 🔥PyTorch | Число выходных нейронов | 
|-|--------|--------|--------|--------|--------|--------|
|1| Бинарная классификация       | Бинарная кросс-энтропия                                 | `torch.nn.BCELoss()`           | Сигмоида  | `torch.nn.Sigmoid()` | 1 |
|2*| Бинарная классификация       | Бинарная кросс-энтропия __без активации последнего нейрона__| `torch.nn.BCEWithLogitsLoss()` | ➖ | ➖ | 1 |
|3*| Многоклассовая классификация | Категориальная кросс-энтропия                           | `torch.nn.CrossEntropyLoss()`  | ➖ | ➖ | Совпадает с числом классов |
|4| Регрессия | Среднеквадратическая ошибка                           | `torch.nn.MSELoss()`  | ➖ | ➖ | 1 |
''')

st.markdown('''

#### Случай № 1
```python

model = nn.Sequential(
    nn.Linear(n, m),
    nn.Sigmoid(),
    nn.Dropout(),
    nn.Linear(m, 1),
    nn.Sigmoid()
)

predictions = model(X) # числа в интервале [0; 1]

loss = torch.nn.BCELoss(predictions, target)
loss.backward()
```

#### Случай № 2

Часто для оптимизации вычислений в задаче бинарной классификации не активируют выходной слой сигмоидой, тогда необходимо использовать в качестве функции потерь `torch.nn.BCEWithLogitsLoss`, однако для получения распределений __вероятностей__ принадлежности объекта классу все равно нужно будет применять сигмоиду. 

```python


model = nn.Sequential(
    nn.Linear(n, m),
    nn.Sigmoid(),
    nn.Dropout(),
    nn.Linear(m, 1)
)

predictions = model(X) 
loss = torch.nn.BCEWithLogitsLoss(predictions, target)
loss.backward()

# получаем вероятности принадлежности объекта классу
probabilities = torch.functional.sigmoid(predictions) # числа в интервале [0; 1]

```
#### Случай № 3
При решении задачи многоклассовой классификации функцию софтмакса мы применяем __только__ для того, чтобы получить вероятностное распределение между классами. В функцию потерь мы передаём «сырые» значения с выходного слоя (т.е. не активированные) – __логиты__. Функция `torch.nn.CrossEntropyLoss()` ожидает на вход именно их, а не числа в интервале $$[0; 1]$$. 


```python
# K - число классов
model = nn.Sequential(
    nn.Linear(n, m),
    nn.Sigmoid(),
    nn.Dropout(),
    nn.Linear(m, K)
)

predictions = model(X) # ⬅️ логиты – любые числа в интервале [-∞; +∞]
loss = torch.nn.CrossEntropyLoss(predictions, target)
loss.backward()

# получаем вероятности принадлежности объекта классу
probabilities = torch.functional.softmax(predictions) # числа в интервале [0; 1]

```
''')

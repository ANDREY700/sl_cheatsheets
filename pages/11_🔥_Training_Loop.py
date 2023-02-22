import streamlit as st
import streamlit.components.v1 as components



components.html(
    """
    <!-- Yandex.Metrika counter -->
<script type="text/javascript" >
   (function(m,e,t,r,i,k,a){m[i]=m[i]||function(){(m[i].a=m[i].a||[]).push(arguments)};
   m[i].l=1*new Date();
   for (var j = 0; j < document.scripts.length; j++) {if (document.scripts[j].src === r) { return; }}
   k=e.createElement(t),a=e.getElementsByTagName(t)[0],k.async=1,k.src=r,a.parentNode.insertBefore(k,a)})
   (window, document, "script", "https://mc.yandex.ru/metrika/tag.js", "ym");

   ym(92504528, "init", {
        clickmap:true,
        trackLinks:true,
        accurateTrackBounce:true,
        webvisor:true
   });
</script>
<noscript><div><img src="https://mc.yandex.ru/watch/92504528" style="position:absolute; left:-9999px;" alt="" /></div></noscript>
<!-- /Yandex.Metrika counter -->
""")

'''

# Обучающий цикл 🔥PyTorch

Оригинал: https://sebastianraschka.com/faq/docs/training-loop-in-pytorch.html

Предположим,  мы хотим обучить нейронную сеть для решения задачи обучения с учителем – классификации или регрессии. 

Обучающий цикл для модели, заданной в [PyTorch](https://pytorch.org), будет выглядеть так: 

```python
model = Model(...)
optimizer = torch.optim.SGD(
  model.parameters(), lr=0.01, momentum=0.9
)

for epoch in range(num_epochs):
    for batch_idx, (features, targets) in enumerate(train_loader):
            
        forward_pass_outputs = model(features)
        loss = loss_fn(forward_pass_outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

В псевдокоде выше мы задаем два цикла `for` и применяем алгоритм обратного распространения ошибки со стохастическим градиентным спуском для оптимизации параметров. 

Вложенный цикл (`for batch_idx ...`) задает обучающий шаг (_training step_), который часто называют _итерацией_. На каждой итерации мы «забираем» из датасета (поверх которого работает загрузчик данных – он оптимизирует процесс) два тензора: тензор с обучающими объектами (`features`) и тензор из лейблов (меток класса или настоящих значений для регрессии: `y_true, target`). 

Далее `model(features)` выполняет прямой проход: прогоняет поданные на вход объекты через нашу модель (а также строит [граф вычислений](https://elbrus-ds-cheatsheets.streamlit.app/Visualization)). В этом примере `model` – произвольная нейронная сеть, решающая задачу обучения с учителем, соответственно, `forward_pass_outputs` – это может быть и одно значение вероятности/логита для задачи бинарной классификации, и распределение вероятностей/логитов по классам в случае мультиклассовой классификации, и просто единственное непрерывное значение в случае регрессии. 

С помощью `loss_fn` (часто функцию потерь именуют как `criterion`), мы вычисляем _разницу_ между предсказанными значениями (`forward_pass_outputs`) и целевыми значениями (`targets`). Эти вычисления также добавляются в граф вычислений. Функция потерь тоже зависит от [задачи](https://elbrus-ds-cheatsheets.streamlit.app/Architecture_Details) – общий ее смысл остается прежним: мы хотим ее минимизировать, чтобы предсказанные значения были как можно «ближе» (в определенном смысле этого слова) к истинным (`targets`).  

Следующие строки выполняют оптимизационную часть, т.е. производят вычисления, являющиеся основой [обратного распространения ошибки](https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%BE%D0%B1%D1%80%D0%B0%D1%82%D0%BD%D0%BE%D0%B3%D0%BE_%D1%80%D0%B0%D1%81%D0%BF%D1%80%D0%BE%D1%81%D1%82%D1%80%D0%B0%D0%BD%D0%B5%D0%BD%D0%B8%D1%8F_%D0%BE%D1%88%D0%B8%D0%B1%D0%BA%D0%B8): 

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

`optimizer` – объект PyTorch, содержащий в себе реализацию конкретного алгоритма оптимизации (градиентный спуск довольно гибкий метод), например стохастического градиентного спуска, ADAM, Adadelta и пр. 

Когда мы запускаем алгоритм обратного распространения ошибки, PyTorch позволяет нам  накапливать значения градиентов – это может быть полезно для некоторых задач, но для нашей задачи это избыточно. Раз по умолчанию градиенты накапливаются, то нам необходимо вручную их обнулить: делается это с помощью метода `zero_grad()`: `optimizer.zero_grad()` (иногда можно встретить `model.zero_grad()` – это то же самое). 

Вызов `loss.backward()` запустит процедуру автоматического дифференцирования (конкретно [reverse mode automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)) на построенном графе вычислений, которая вычислит необходимые градиенты по параметрам нашей сети, которые мы хотим скорректировать для получения более точных результатов. Обновление параметров производится в момент вызова `optimizer.step()`. 

Далее, после выполнения всех итераций вложенного цикла, внешний цикл повторяет эту процедуру множество эпох. 

> Эпоха – название для одного цикла обучения нейронной сети, в котором все данные обучающей выборки **один раз** «прошли» через нейросеть, повлияли на вычисленные градиенты и, как следствие, на параметры сети. Нейронные сети, как правило, обучаются на множестве эпох. 

### Заключение

Обычный обучающий цикл (_training loop_) в PyTorch итерируется по батчам («пачкам» объектов из обучающей выборки) заданное количество эпох. В каждой итерации (для каждого батча) мы получаем выходы модели и вычисляем значение функции потерь между выходами модели (предсказаниями) и целевой переменной (`targets`). 

```python
forward_pass_outputs = model(features)
loss = loss_fn(forward_pass_outputs, targets)
```

Далее мы обнуляем накопленные ранее градиенты (это особенность PyTorch) и вычисляем градиенты по параметрам сети: 

```python
optimizer.zero_grad()
loss.backward()
```

В конце обновляем значения параметров модели с учетом вычисленных градиентов с помощью заданного алгоритма оптимизации: 

```python
optimizer.step()
```


'''
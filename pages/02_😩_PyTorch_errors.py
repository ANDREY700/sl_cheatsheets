from cgitb import text
from tkinter.messagebox import QUESTION
import streamlit as st

from aux.render_block import Block
            
        

st.title('`pytorch` errors')

st.info('Используй `ctrl+f` для поиска на этой странице')
"""---"""

Block(
    # 'Несоотвествие размерностей векторов',
    'Чаще всего в вычислении лосс-функции в задаче бинарной классификации или регрессии',
    'ValueError: Using a target size (torch.Size([N, 1])) that is different to the input size (torch.Size([N])) is deprecated. Please ensure they have the same size.',
    'Приводим к вектора к одинакову размеру.',
    'loss = criterion(y_pred.squeeze(-1), y_true)',
).render_block()

Block(
    # 'Несоотвествие размерностей слоев и размерностей данных',
    'Почти наверняка архитектура модели некорректна',
    'RuntimeError: mat1 and mat2 shapes cannot be multiplied (IxJ and NxM)',
    'Проверяем архитектуру модели, где-то не совпадают значения входов и выходов разных слоёв, либо данные не соответствуют размерам.',
    '''
...
nn.Linear(128, 10), 👈 10 нейронов 
nn.Sigmoid(),
nn.Linear(16, 1) 👈 16 нейронов
...
    '''
).render_block()

Block(
    # 'Несоответствие в числе каналов в свёрточных слоях',
    'Какой-то из сверточных слоёв не ожидает числа feature maps, которое ему передает предыдущий слой. ',
    'RuntimeError: Given groups=1, weight[n, c, h, w], so expected input[p, q, x, y] to have w channels, but got 64 channels instead',
    'Проверяем архитектуру модели, где-то не совпадают значения входов и выходов разных сврточных слоёв (скорее всего число каналов отличается).',
'''
...
nn.Conv2d(3, 64, kernel_size=5), 👈 64 feature maps на выходе
nn.ReLU(),
nn.LazyBatchNorm2d(),
nn.Conv2d(32, 16, kernel_size=3), 👈 32 feature maps на входе
...
    '''
).render_block()



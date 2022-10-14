import streamlit as st

'''
# Подготовка данных для моделей 🔥PyTorch

Для подгтовки данных в PyTorch используются специальные классы. Если данных мало, 
то можно просто конвертировать их в тензор и подавать на вход модели, но если необходимо разбивать данные на 
батчи, то нужно использовать `DataLoader`. 

## Табличные данные

### Простой вариант (не рекомендуется)

```python
from sklearn.datasets import make_classification
X, y = make_classification()
print(f'Types: {type(X)}, {type(y)}')

Types: <class 'numpy.ndarray'>, <class 'numpy.ndarray'>

model = nn.Sequential(
    nn.Linear(20, 32),
    nn.Sigmoid(),
    nn.Linear(32, 1)
)

model(X)

TypeError: linear(): argument 'input' (position 1) must be Tensor, 
           not numpy.ndarray
``` 

Конвертировать данные в тензоры можно так: 

```python
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

y_pred = model(X)
print(y_pred)

tensor([[0.0214],
        [0.0659],
        [0.1001]], grad_fn=<SliceBackward0>)
```

### TensorDataset

Для формирования объекта датасета нужно использовать класс TensorDataset: 

```python
from torch.utils.data import TensorDataset

dataset = TensorDataset(
    torch.from_numpy(X).type(torch.float32), 
    torch.from_numpy(y).type(torch.float32)
)
``` 

Если нужно разбить выборку на обучающую и валидационную, то можно 
воспользоваться функцией `torch.utils.data.random_split`: 

```python
train_ds, valid_ds = torch.utils.data.random_split(train_dataset, [70, 30])
```

Теперь можно передавать датасеты в `DataLoader`:

```python
from torch.utils.data import TensorDataset, DataLoader

train_loader = DataLoader(train_ds, shuffle=True, batch_size=64)
valid_loader = DataLoader(valid_ds, shuffle=True, batch_size=64)
```

## Изображения

Для изображений в `torchvision.datasets` есть класс `ImageFolder`. Для корректной работы нужна следующая структура: 
'''
st.code('''
📂data
|--📂train
|----📂class1
|------🖼img1.png
|------🖼img2.png
|------ ...
|----📂class2
|------🖼img1.png
|------🖼img2.png
|------ ...
|--📂valid
|----📂class1
|------🖼img1.png
|------🖼img2.png
|------ ...
|----📂class2
|------🖼img1.png
|------🖼img2.png
|------ ...

'''
)
'''

```python
from torchvision.datasets import ImageFolder

train_dataset = ImageFolder('data/train', transform=augmentations)

print(train_dataset.class_to_idx)
> {'class1': 0, 'class2': 1}

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
```

Число классов `ImageFolder` определит по числу папок в директориях `train`, `valid` и т.д.
Он упорядочит их по имени и назначит метки классов от 0 до N-1 (где N – число классов). 

Часто `ImageFolder` не удовлетворяет нашим потребностям. Например, для обучения Denoising Autoencoder нам нужно забирать 
из папок пары картинок: чистую и зашумленную версию. Ниже приведена базовая версия класса `CustomImageDataset`:

```python
import os
from torchvision import transforms as T
from torchvision.io import read_image

preprocessing = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((100, 200)), # <------ свой размер
        T.ToTensor()
    ]
)

class CustomImageDataset(Dataset):
    def __init__(self, noise_dir, clean_dir, aug=None):
        self.noise_dir = noise_dir
        self.clean_dir = clean_dir
        self.noise_names = sorted(os.listdir(noise_dir))
        self.clean_names = sorted(os.listdir(clean_dir))
        self.aug = aug
    def __len__(self):
        return len(self.noise_names)
    


    def __getitem__(self, idx):
        noisy_img = read_image(os.path.join(self.noise_dir, self.noise_names[idx]))
        clean_img = read_image(os.path.join(self.clean_dir, self.clean_names[idx]))
        if self.aug:
            noisy_img = self.aug(noisy_img)
            clean_img = self.aug(clean_img)
        return noisy_img, clean_img
```
'''



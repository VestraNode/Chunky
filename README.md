# Chunky — адаптивна стеганографія в графічних контейнерах

**Chunky** — бібліотека мовою Python для приховування тексту й довільних байтів у цифрових зображеннях. На відміну від класичного методу молодших бітів (LSB), який розкидає зміни по всьому фото, Chunky записує дані переважно в найшумніші ділянки зображення, де зміни губляться серед природного шуму.

Це основа проєкту **«Створення програмного комплексу адаптивної стеганографії в графічних контейнерах та його інтеграція у веб-застосунок VestraNode»**.

- Сайт і веб-інтерфейс: **https://vn.raimex.space**
- Автори: Raimex Team, Подгорський Костянтин Олександрович

## Дві версії методу

| Версія | Ідея |
|---|---|
| `SChunky` | Еталонна версія: фото ділиться на чанки, дані записуються від найшумніших чанків до найтихіших, номер-маркер у чанку дозволяє декодеру самостійно зібрати повідомлення. |
| `ChunkyV1.1` | Розвиток: маркер лише в молодших бітах, біти записуються зі зміною значення на одиницю в бік меншого шуму (LSB matching), кодер перевіряє результат власним декодуванням. |

## Як це працює (просто, по кроках)

1. **Підготовка.** Текст (або байти) за потреби архівується (zlib, bz2 або lzma, обирається найменший результат), додається службовий заголовок.
2. **Оцінка шуму.** Для кожного пікселя береться прогноз за чотирма сусідами; чим більше значення відхиляється від прогнозу, тим «шумніший» піксель.
3. **Чанки.** Фото ділиться на квадратні чанки розміром `x × x` (за замовчуванням 8×8). Рівень шуму чанка — кількість «шумних» каналів (R, G, B) його пікселів. Чанки сортуються від найшумніших до найтихіших.
4. **Запис.** Біти даних записуються в чанки в цьому порядку (до `y` пікселів на чанк). У кожного чанка є номер-маркер, що вказує його місце в повідомленні. З ключем позиції в чанку перемішуються.
5. **Версія ChunkyV1.1.** Якщо значення вже збігається з потрібним бітом, воно не змінюється; інакше змінюється на ±1 у той бік, де шум стає меншим. Після запису кодер сам декодує результат і перевіряє, що повідомлення відновлюється.
6. **Декодування.** Декодер заново рахує шум і рейтинг чанків, знаходить чанки за маркерами, збирає біти, розпаковує дані. Розмір чанка `x` він добирає самостійно.

## Встановлення

```bash
git clone https://github.com/VestraNode/Chunky.git
cd Chunky
pip install -r requirements.txt
```

Інтернет потрібен лише для встановлення залежностей; сама бібліотека працює повністю офлайн, дані не залишають вашого пристрою.

## Використання

Командний рядок:

```bash
python example.py hide photo.jpg "Секретне повідомлення" result.png
python example.py reveal result.png

python example.py hide photo.jpg "Текст" result.png --method schunky --key 42
python example.py reveal result.png --method schunky --key 42
```

Як бібліотека:

```python
import numpy as np
from PIL import Image
from stego.chunky11 import encoder, decoder

image = np.array(Image.open("photo.jpg").convert("RGB"))
result, report = encoder.encode_with_report(image, 8, 40, "Привіт".encode("utf-8"))
Image.fromarray(result).save("result.png")

text, info = decoder.decode_with_report(np.array(Image.open("result.png").convert("RGB")))
print(text.decode("utf-8"))
```

Параметри кодера `encode_with_report(image, x, y, data, use_key=False, key=0, threshold=8, compress=True)`:

- `x` — розмір чанка (пікселів);
- `y` — максимум пікселів, що використовуються в чанку для даних;
- `use_key`, `key` — перемішування позицій у чанку за числовим ключем;
- `threshold` — поріг шуму;
- `compress` — архівувати дані перед записом.

Звіт `report` містить кількість використаних чанків, розмір даних після архівації та інші показники.

## Склад репозиторію

| Файл | Призначення |
|---|---|
| `stego/SChunky/` | кодер і декодер версії SChunky |
| `stego/ChunkyV1.1/` | кодер, декодер, розкладка й параметри версії ChunkyV1.1 |
| `stego/analysis.py` | оцінка шуму пікселів (стегоаналіз) |
| `stego/chunking.py` | чанки та їх рейтинг за шумом |
| `stego/common.py` | спільні функції: біти, маркери, перемішування за ключем |
| `stego/compression.py` | вибір найкращої архівації |
| `stego/lsb_match.py` | запис бітів методом LSB matching |
| `stego/stats.py` | показники змін зображення (PSNR, кількість змінених значень) |
| `stego/steg_views.py` | карти стегоаналізу для візуалізації |
| `example.py` | приклад командного рядка |

## Ліцензія

MIT. Повний текст у файлі [LICENSE](LICENSE). Автор: Raimex Team, Подгорський Костянтин Олександрович.

## Обмеження

- Контейнер має бути без втрат (PNG) або зображенням, яке не буде стискатися повторно. Перекодування в JPEG руйнує приховані біти.
- Розмір повідомлення обмежений ємністю фото: чим більше шумних ділянок, тим більше даних вміщується.
- Метод ускладнює виявлення за статистичними ознаками, але не гарантує невидимості проти будь-якого стегоаналізатора.

---

## English summary

**Chunky** is a Python library for adaptive image steganography. It splits an image into square chunks, estimates the noise of each chunk (pixel vs. prediction from its four neighbours) and writes the hidden bits into the noisiest chunks first, so changes are masked by natural noise. Two versions are included: `SChunky` (reference) and `ChunkyV1.1` (LSB matching with ±1 changes toward lower noise and a built-in self-check). Use lossless containers (PNG); JPEG re-compression destroys the payload. Everything runs offline. Web UI: https://vn.raimex.space

Authors: Raimex Team, Kostiantyn Podhorskyi. License: MIT.

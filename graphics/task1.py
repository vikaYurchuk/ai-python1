import numpy as np
import matplotlib.pyplot as plt

# Завдання 1
x = np.linspace(-10, 10, 400)
# y = np.sin(x)
y = x**2 * np.sin(x)

plt.plot(x, y)
plt.title("Графік f(x) = x^2 * sin(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

#Завдання 2

data = np.random.normal(loc=5, scale=2, size=1000)

plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title("Гістограма нормального розподілу")
plt.xlabel("Значення")
plt.ylabel("Частота")
plt.grid(True)
plt.show()

#Завдання 3

labels = ['Читання', 'Спорт', 'Танці', 'Музика', 'Подорожі']
sizes = [10, 20, 35, 20, 25]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Мої улюблені хобі")
plt.axis('equal') 
plt.show()

#Завдання 4

fruits = {
    "Банани": np.random.normal(150,10,100),
    "Сливи": np.random.normal(120,8,100),
    "Персики": np.random.normal(140,9,100),
    "Мандарини": np.random.normal(130,12,100),
}
plt.boxplot(fruits.values(),labels=fruits.keys())
plt.title("Маси фруктів")
plt.ylabel("Маса(г)")
plt.grid(True)
plt.show()


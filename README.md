## 🧠 Neural Network from Scratch
Prosta implementacja sieci neuronowej napisana od podstaw w Pythonie przy użyciu biblioteki NumPy. 
Projekt został stworzony w celach edukacyjnych, aby krok po kroku zrozumieć, jak działa sieć neuronowa.
Pomysł zaczerpnięty z wielu filmików na youtube wyjaśniających podane zagadnienie.

Cała implementacja została napisana bez użycia frameworków typu TensorFlow czy PyTorch, dzięki czemu każdy element — od propagacji w przód (forward propagation), przez obliczanie błędu, aż po backpropagation i aktualizację wag — jest jawnie zaimplementowany w kodzie.

## 📌 Funkcjonalności
- Implementacja warstw neuronowych (NeuronLayer)
- Obsługa funkcji aktywacji
- Forward propagation
- Backpropagation (ręcznie liczony gradient)
- One-hot encoding
- Aktualizacja wag metodą gradient descent
- Obliczanie accuracy
- Trenowanie na danych MNIST

## 🏗️ Architektura sieci
Sieć składa się z 3 warstw:
1. Wejście: 784 neurony (28x28 pikseli)
2. Warstwa ukryta 1: 64 neurony (ReLU)
3. Warstwa ukryta 2: 32 neurony (ReLU)
4. Wyjście: 10 neuronów (Softmax)

## 📊 Skuteczność modelu
Po przeprowadzeniu treningu na zbiorze MNIST model osiąga około **95% dokładności** na danych treningowych.

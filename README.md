# CV2

На основе ноутбука из лекции, настроить и обучить полносвязную прямую нейронную сеть классифицирвоать объекты из датасета cifar10 (https://keras.io/api/datasets/cifar10/). Желательно добиться как можно большего значения accuracy. Можно менять функции активации, функцию ошибки (loss), типы слоев, количество нейронов в слоях, количество слоев, оптимизаторы, количество эпох и т.д. Нельзя менять метрику и тип нейронной сети.

Установка слоёв модели, а так же компиляция
```python
model = keras.Sequential([
                          keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                          keras.layers.MaxPooling2D(pool_size=(2, 2)),
                          keras.layers.Conv2D(64, 3, activation="relu"),
                          keras.layers.MaxPooling2D(pool_size=(2, 2)),
                          keras.layers.Conv2D(128, 3, activation="relu"),
                          keras.layers.Flatten(),
                          keras.layers.Dense(128, activation='relu'),
                          keras.layers.Dense(20, activation="softmax")
])
```
```python
model.compile(optimizer=tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
```python
model.summary()
```
<code>
Model: "sequential_10"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_21 (Conv2D)          (None, 30, 30, 32)        896       
                                                                 
 max_pooling2d_14 (MaxPoolin  (None, 15, 15, 32)       0         
 g2D)                                                            
                                                                 
 conv2d_22 (Conv2D)          (None, 13, 13, 64)        18496     
                                                                 
 max_pooling2d_15 (MaxPoolin  (None, 6, 6, 64)         0         
 g2D)                                                            
                                                                 
 conv2d_23 (Conv2D)          (None, 4, 4, 128)         73856     
                                                                 
 flatten_10 (Flatten)        (None, 2048)              0         
                                                                 
 dense_20 (Dense)            (None, 128)               262272    
                                                                 
 dense_21 (Dense)            (None, 20)                2580      
                                                                 
=================================================================
Total params: 358,100
Trainable params: 358,100
Non-trainable params: 0
</code>

Обучение модели
```python
model.fit(x_train, y_train, epochs=25)
```
<code>
Epoch 1/25
1563/1563 [==============================] - 89s 56ms/step - loss: 2.0717 - accuracy: 0.2416
Epoch 2/25
1563/1563 [==============================] - 86s 55ms/step - loss: 1.6947 - accuracy: 0.3892
Epoch 3/25
1563/1563 [==============================] - 87s 55ms/step - loss: 1.5192 - accuracy: 0.4550
Epoch 4/25
1563/1563 [==============================] - 85s 54ms/step - loss: 1.3934 - accuracy: 0.5017
Epoch 5/25
1563/1563 [==============================] - 90s 58ms/step - loss: 1.2926 - accuracy: 0.5417
Epoch 6/25
1563/1563 [==============================] - 85s 54ms/step - loss: 1.2109 - accuracy: 0.5726
Epoch 7/25
1563/1563 [==============================] - 87s 55ms/step - loss: 1.1367 - accuracy: 0.5987
Epoch 8/25
1563/1563 [==============================] - 85s 54ms/step - loss: 1.0706 - accuracy: 0.6247
Epoch 9/25
1563/1563 [==============================] - 87s 56ms/step - loss: 1.0149 - accuracy: 0.6469
Epoch 10/25
1563/1563 [==============================] - 86s 55ms/step - loss: 0.9610 - accuracy: 0.6656
Epoch 11/25
1563/1563 [==============================] - 85s 54ms/step - loss: 0.9130 - accuracy: 0.6812
Epoch 12/25
1563/1563 [==============================] - 88s 56ms/step - loss: 0.8709 - accuracy: 0.6975
Epoch 13/25
1563/1563 [==============================] - 85s 54ms/step - loss: 0.8267 - accuracy: 0.7119
Epoch 14/25
1563/1563 [==============================] - 87s 56ms/step - loss: 0.7836 - accuracy: 0.7271
Epoch 15/25
1563/1563 [==============================] - 86s 55ms/step - loss: 0.7453 - accuracy: 0.7421
Epoch 16/25
1563/1563 [==============================] - 86s 55ms/step - loss: 0.7082 - accuracy: 0.7543
Epoch 17/25
1563/1563 [==============================] - 85s 54ms/step - loss: 0.6691 - accuracy: 0.7679
Epoch 18/25
1563/1563 [==============================] - 86s 55ms/step - loss: 0.6315 - accuracy: 0.7801
Epoch 19/25
1563/1563 [==============================] - 87s 56ms/step - loss: 0.6004 - accuracy: 0.7900
Epoch 20/25
1563/1563 [==============================] - 87s 56ms/step - loss: 0.5657 - accuracy: 0.8030
Epoch 21/25
1563/1563 [==============================] - 85s 54ms/step - loss: 0.5298 - accuracy: 0.8172
Epoch 22/25
1563/1563 [==============================] - 87s 55ms/step - loss: 0.4945 - accuracy: 0.8285
Epoch 23/25
1563/1563 [==============================] - 86s 55ms/step - loss: 0.4643 - accuracy: 0.8375
Epoch 24/25
1563/1563 [==============================] - 85s 54ms/step - loss: 0.4288 - accuracy: 0.8498
Epoch 25/25
1563/1563 [==============================] - 88s 56ms/step - loss: 0.3932 - accuracy: 0.8624
</code>

Результат обучения
```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
```
<code>
313/313 [==============================] - 5s 15ms/step - loss: 0.9980 - accuracy: 0.7023
Test loss: 0.9979968070983887
Test accuracy: 0.7023000121116638
</code>

![](https://github.com/SAJQDQ/CV2/blob/main/CV2%20png/Screenshot_3.png)

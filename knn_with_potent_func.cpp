//Метод потенциальных функций
//potential function method

#include <iostream>
#include <vector>
#include <cmath>

class KNN_kernel {
public:
  KNN_kernel(int k = 3) : k(k), accur(100) {}
  KNN_kernel(std::vector <std::vector <double>> vectors,
	     std::vector <int> classes, int k = 3) : vectors(vectors), classes(classes), 
						     k(k){
    for (size_t i = 0; i < vectors.size(); i++) {
      potentual.push_back(0); //Потенциал пока равен нулю
    }
    accur = 100;
  }
  ~KNN_kernel() {}
  //Добавление новых векторов
  void fit(std::vector <std::vector <double>> vec,
	     std::vector <int> cls) {
    //Вставляем новые вектора в уже имеющиеся
    vectors.insert(vectors.end(), vec.begin(), vec.end());
    classes.insert(classes.end(), cls.begin(), cls.end());
    for (size_t i = 0; i < vec.size(); i++) {
      potentual.push_back(0); //Потенциал пока равен нулю
    }
    std::vector <int> accuracy(vectors.size()); //1 - ошибка, 0 - все ок.
    //Настраиваем потенциалы
    while (accur > 60) {
      //По очереди обходим всю выборку и предстказыаем каждый вектор
      for (size_t i = 0; i < vectors.size(); i++) {
	int pred_cls = algo(vectors[i]);
	if (pred_cls != classes[i]) {	  
	  potentual[i] += 1;	  
	  accuracy[i] = 1;
	} else {
	  accuracy[i] = 0;
	}
      }
      //Считаем количество ошибок
      int new_accur = 0;
      for (size_t i = 0; i < accuracy.size(); i++) {
	new_accur += accuracy[i];
      }
      //Выводим средний процент ошибок
      accur = (new_accur * 100) / accuracy.size();
      //Нужна для обратой связи, но лучше убрать.
      std::cout << "Mean accuracy = " << accur << std::endl;
    }    
  }
  //Предсказание
  int predict(std::vector <double> vector) {    
    return algo(vector);    
  }

private:
  int k; //Количество соседей
  std::vector <std::vector <double>> vectors; //Вектора признаков
  std::vector <int> classes; //Классы векторов
  bool fited; //Нужно ли извлекать уникальные классы
  std::vector <int> potentual; //Потенциал вектора
  double accur; //Велечина ошибки

  //Эвклидова метрика
  double p(std::vector <double> vector, std::vector <double> base_vector) {
    double dist = 0;
    size_t vec_size = vector.size();
    for (size_t i = 0; i < vec_size; i++) {
      dist += pow(base_vector[i] - vector[i], 2);
    }
    return sqrt(dist);
  }
  //Сортировка методом пузырька
  void sort( std::vector <double> & distance) {    
    for (int i = 0; i < distance.size() - 1; i++) {
      for (int j = 0; j < distance.size() - i - 1; j++) {
	if (distance[j] > distance[j + 1]) {
	  // меняем элементы местами
	  std::swap(distance[j], distance[j + 1]);	  
	  //Меняем вектора в массиве так же местами
	  std::swap(vectors[j], vectors[j+1]);
	  std::swap(classes[j], classes[j+1]);
	  std::swap(potentual[j], potentual[j+1]);
	}
      }
    }
  }
  //Ядро
  double kernel(double x) {
    return  (x != 0 ? 1 / x: 0);
  }
  //Максимальный элемент
  int arg_max(const std::vector <double> & arr) {
    int max = -1;
    int elem_num = 0;
    for (size_t i = 0; i < arr.size(); i++) {
      if (max < arr[i]) {
	max = arr[i];
	elem_num = i;
      }
    }
    return elem_num;
  }
  //Уникальные значения
  std::vector <int> uniq(std::vector <int> & classes) {
    bool in_arr = false;
    std::vector <int> classes2;
    size_t cls_size = classes.size();
    for (size_t i = 0; i < cls_size; i++) {
      std::vector<int>::iterator it = classes2.begin();
 
      while(it != classes2.end()) {
	if (*it == classes[i]) {
	  in_arr = true;
	  break;
	}
	it++;
      }
      if (in_arr == false) {
	classes2.push_back(classes[i]);	
      } else {
	in_arr = false;
      }
    }
    return classes2;
  }

  int algo(std::vector <double> & vector) {
    //Считаем расстояние от входного вектора до каждого из базы векторов
    std::vector <double> distance;
    size_t v_size = vectors.size();
    for (size_t i = 0; i < v_size; i++) {
      distance.push_back(p(vector, vectors[i]));
    }
    //Сортируем
    sort(distance);    
    //Получаем уникальные значения классов
    std::vector <int> cls_unique = uniq(classes);    
    //Вычисляем значения суммирующей функции для каждого класса
    size_t counter = k <= distance.size() ? k: distance.size(); //
    std::vector <double> arr(cls_unique.size());
    for (size_t i = 0; i < cls_unique.size(); i++) {
      for (size_t j = 0; j < counter; j++) {
	arr[i] += (classes[j] == cls_unique[i] ? 1: 0) * potentual[i] * \
	  kernel (distance[j]/ distance[j+1]);
      }
    }    
    int max_cls = arg_max(arr);
    return cls_unique[max_cls];    
  }
};

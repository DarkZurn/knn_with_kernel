#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

//Интерфейс нейрона
class INeuron {
public:
  virtual ~INeuron() = default;
  virtual double predict(boost::numeric::ublas::matrix<double> inputMatrix) = 0;
  virtual double predict(boost::numeric::ublas::vector<double> inputVector) = 0;
};

//Перцептрон розенблата
class Perceptron : public INeuron {
public:
  Perceptron(int n) : input_num(n) {
    weights = new boost::numeric::ublas::matrix<double>(n, 1);
    for(size_t i = 0; i < weights->size1(); ++i) {
      for(size_t j = 0; j < weights->size2(); ++j) {
	(*weights)(i, j) = (double)rand() / (double)RAND_MAX;
      }
    }
  }
  ~Perceptron() {
    delete weights;
  }
  virtual double predict(boost::numeric::ublas::matrix<double> inputMatrix) override {
    return (((prod(inputMatrix, *weights)(0, 0)) + bias) > 0 ? 1 : 0);
  }
  virtual double predict(boost::numeric::ublas::vector<double> inputVector) override{
    return (((prod(inputVector, *weights)(0)) + bias) > 0 ? 1 : 0);
  }
  //Обучение перцептрона на единственном примере
  int train_on_single_example(boost::numeric::ublas::matrix<double> input, int y) {
    int error = y - (((prod(input, *weights)(0, 0)) + bias) > 0 ? 1 : 0);
    if (error != 0) {
      //Подправляем веса
      input = input * error;     
      *weights += trans(input);
      bias += error;
    }    
    return error;
  }
  int train_on_single_example(boost::numeric::ublas::vector<double> input, int y) {    
    int error = y - (((prod(input, *weights)(0)) + bias) > 0 ? 1 : 0);
    if (error != 0) {
      //Подправляем веса
      input = input * error;     
      for(size_t i = 0; i < weights->size1(); ++i) {
	(*weights)(i, 0) = input(i);
      }
      bias += error;
    }    
    return error;
  }
  //Обучение на всей порции примеров до сходимости
  bool train_until_convergence(boost::numeric::ublas::matrix<double> input,
			       boost::numeric::ublas::matrix<double> y, size_t max_train) {
    bool perfect = false;
    size_t i = 0;
    while((perfect == false) && (i++ < max_train)) {
      perfect = true;
      //Обучаем
      for(size_t i = 0; i < input.size1(); ++i) {
	boost::numeric::ublas::vector<double> sub_vec (input.size2());
	for(size_t j = 0; j < input.size2(); ++j) {
	  sub_vec(j) = input(i, j);	  
	}
	int error = this->train_on_single_example(sub_vec, y(i, 0));
	if (error != 0) {
	  perfect = false;
	}
      }          
    }
    return perfect;
  }
private:
  int input_num; //Количество входов
  boost::numeric::ublas::matrix<double> *weights; //Веса входных связей
  double bias = 1; //Смещение
};

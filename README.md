# knn_with_kernel

Parzenovsky window method.


Usage:

int main() {
  KNN_kernel *clf = new KNN_kernel();

  std::vector <std::vector <double>> vectors;
  std::vector <int> classes;
  
  //Заполняем
  for (size_t i = 0; i < 10; i++) {
    vectors.push_back(std::vector <double>());
    for (size_t j = 0; j < 10; j++) {      
      if ((i+j) %2 == 0) {
	vectors[i].push_back((double)((int)i+(int)j - 1));
      } else {
	vectors[i].push_back((double)((int)i-(int)j + 1));
      }      
    }
    if (i %2 == 0) {
      classes.push_back(0);
    } else {
      classes.push_back(1);
    }
  }
 
  clf->fit(vectors, classes);
  std::cout << "Predicted class = " << clf->predict(vectors[0]) << std::endl;

  
  delete clf;
  std::cout << "Done" << std::endl;
}

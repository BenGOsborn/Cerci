#include <iostream>
#include <vector>

class Matrix {
    public:
        Matrix(std::vector<float> *matrix) {
            matrix_raw = {};
            matrix_raw->push_back(*matrix);
            // *rows = 1;
            // *cols = matrix->size();
        }

        Matrix(std::vector<std::vector<float>> *matrix) {
            *matrix_raw = *matrix;
            *rows = matrix->size();
            *cols = matrix->at(0).size();
        }

        Matrix(std::vector<int> dims, float(*func)()) {
            *rows = dims[0];
            *cols = dims[1];

            std::vector<float> tempArray;
            float val;
            for (int row=0; row < dims[0]; row++) {
                tempArray = {};
                for (int col=0; col < dims[1]; col++) {
                    val = func();
                    tempArray.push_back(val);
                }
                matrix_raw->push_back(tempArray);
            }
        }

        void print() {
            for (int row; row < *rows; row++) {
                for (int col; col< *cols; col++) {
                    std::cout << matrix_raw->at(row)[col] << " ";
                }
                std::cout << "\n";
            }
        }

        std::vector<int> size() {
            std::vector<int> size = {*rows, *cols};
            return size;
        }

    private:
        std::vector<std::vector<float>> *matrix_raw;
        int *rows, *cols;

};

int main() {
    std::vector<float> arr = {1, 2, 3};

    // Why am I getting so many core dumps what is going on man?

    // Gonna have to make sure that the vectors are all of the same lengths though or else is bad
    Matrix *matrix = new Matrix(&arr);
    std::cout << matrix->size()[0] << std::endl;

    return 0;
}
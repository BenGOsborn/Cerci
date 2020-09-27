#include <iostream>
#include <vector>

class Matrix {
    public:
        Matrix(std::vector<float> vec) {
            matrix_raw->push_back(vec);
            rows = new int(1);
            cols = new int(vec.size());
        }

        Matrix(std::vector<std::vector<float>> matrix) {
            // Probably a more efficient way of doing this
            for (int i; i<matrix.size(); i++) {
                matrix_raw->push_back(matrix[i]);
            }
            rows = new int(matrix.size());
            cols = new int(matrix[0].size());
        }

        Matrix(std::vector<int> dims, float(*func)()) {
            rows = new int(dims[0]);
            cols = new int(dims[1]);

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
        std::vector<std::vector<float>> *matrix_raw = new std::vector<std::vector<float>>();
        int *rows, *cols;

};

int main() {
    std::vector<float> arr = {1.0f, 2.0f, 3.0f};

    Matrix *matrix = new Matrix(arr);
    std::cout << matrix->size()[0] << std::endl;

    return 0;
}
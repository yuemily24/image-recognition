#include "dectree.h"

// Makefile included in starter:
//    To compile:               make
//    To decompress dataset:    make datasets

/**
 * main() takes in 2 command line arguments:
 *    - training_data: A binary file containing training image / label data
 *    - testing_data: A binary file containing testing image / label data
 * 
 */
int main(int argc, char *argv[]) {
  int total_correct = 0;

  // parse command line arguments
  Dataset *training_data = load_dataset(argv[1]);
  Dataset *testing_data = load_dataset(argv[2]);

  // build decision tree with training data
  DTNode *training_root = build_dec_tree(training_data);

  // for each test image, compare predicted label and real label
  for (int i = 0; i < testing_data -> num_items; i++) {
    int predicted_label = dec_tree_classify(training_root, &(testing_data -> images[i]));
    int real_label = testing_data -> labels[i];
    if (predicted_label == real_label) { // if labels match, increment 'total_correct' by one
      total_correct += 1;
    }
  }

  // free all dynamically allocated data
  free_dec_tree(training_root);
  free_dataset(training_data);
  free_dataset(testing_data);

  // Print out answer
  printf("%d\n", total_correct);
  return 0;
}

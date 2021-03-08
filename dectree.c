#include "dectree.h"

/**
 * Load the binary file, filename into a Dataset and return a pointer to 
 * the Dataset. The binary file format is as follows:
 *
 *     -   4 bytes : `N`: Number of images / labels in the file
 *     -   1 byte  : Image 1 label
 *     - NUM_PIXELS bytes : Image 1 data (WIDTHxWIDTH)
 *          ...
 *     -   1 byte  : Image N label
 *     - NUM_PIXELS bytes : Image N data (WIDTHxWIDTH)
 *
 */
Dataset *load_dataset(const char *filename) {
    // open binary file
    FILE *data_file = NULL;
    data_file = fopen(filename, "rb");
    // error check if file opened correctly
    if (data_file == NULL) {
        fprintf(stderr, "Error: could not open file\n");
    }

    // allocate memory for a Dataset struct
    Dataset *data_set_ptr = NULL;
    data_set_ptr = malloc(sizeof(Dataset)); 
    // check if memory allocation for data_set_ptr was successful
    if (data_set_ptr == NULL) {
        fprintf(stderr, "Error: memory allocation\n");
    }
    
    // set total number of images in the dataset
    int *total_images = malloc(sizeof(int));
    fread(total_images, sizeof(int), 1, data_file);
    data_set_ptr -> num_items = (*total_images); 
    free(total_images);

    // allocate memory for image and label arrays
    data_set_ptr -> images = malloc(sizeof(Image) * (data_set_ptr -> num_items));
    data_set_ptr -> labels = malloc(sizeof(unsigned char) * (data_set_ptr -> num_items));
    // check if memory allocation was successful
    if (data_set_ptr -> images == NULL || data_set_ptr -> labels == NULL) {
        fprintf(stderr, "Error: memory allocation\n");
    }
    
    // set array variables for data_set_ptr
    for (int i = 0; i < (data_set_ptr -> num_items); i++) {
        // read in image label
        fread((data_set_ptr -> labels) + (i * sizeof(unsigned char)), sizeof(unsigned char), 1, data_file);
        // set struct values and read in image pixel values 
        (data_set_ptr -> images)[i].sx = WIDTH;
        (data_set_ptr -> images)[i].sy = WIDTH;
        (data_set_ptr -> images)[i].data = malloc(sizeof(unsigned char) * (WIDTH * WIDTH));
        fread(data_set_ptr -> images[i].data, sizeof(unsigned char), NUM_PIXELS, data_file);
    }

    // close binary file
    int error = fclose(data_file);
    if (error != 0) {
        fprintf(stderr, "Error: fclose failed\n");
    }

    return data_set_ptr;
}

/**
 * Compute and return the Gini impurity of M images at a given pixel
 * The M images to analyze are identified by the indices array. The M
 * elements of the indices array are indices into data.
 * This is the objective function used to identify the best 
 * pixel on which to split the dataset when building the decision tree.
 *
 * Note that the gini_impurity implemented here can evaluate to NAN 
 * (Not A Number) and will return that value. Ensure that a pixel whose 
 * gini_impurity evaluates to NAN is not used to split the data.  (see find_best_split)
 * 
 */
double gini_impurity(Dataset *data, int M, int *indices, int pixel) {
    int a_freq[10] = {0}, a_count = 0;
    int b_freq[10] = {0}, b_count = 0;

    for (int i = 0; i < M; i++) {
        int img_idx = indices[i];

        // The pixels are always either 0 or 255, but using < 128 for generality.
        if (data->images[img_idx].data[pixel] < 128) {
            a_freq[data->labels[img_idx]]++;
            a_count++;
        } else {
            b_freq[data->labels[img_idx]]++;
            b_count++;
        }
    }

    double a_gini = 0, b_gini = 0;
    for (int i = 0; i < 10; i++) {
        double a_i = ((double)a_freq[i]) / ((double)a_count);
        double b_i = ((double)b_freq[i]) / ((double)b_count);
        a_gini += a_i * (1 - a_i);
        b_gini += b_i * (1 - b_i);
    }

    // Weighted average of gini impurity of children
    return (a_gini * a_count + b_gini * b_count) / M;
}

/**
 * Given a subset of M images and the array of their corresponding indices, 
 * find and use the last two parameters (label and freq) to store the most
 * frequent label in the set and its frequency.
 *
 * - The most frequent label (between 0 and 9) will be stored in `*label`
 * - The frequency of this label within the subset will be stored in `*freq`
 * 
 * If multiple labels have the same maximal frequency, return the smallest one.
 */
void get_most_frequent(Dataset *data, int M, int *indices, int *label, int *freq) {
    // keep an array of the frequency of labels between 0 and 9. for instance, frequency of label 2 can be found at frequencies[2]
    int frequencies[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // loop through the labels of the M subset of images and record them in frequencies
    for (int j = 0; j < M; j++) {
        int img_index = indices[j];
        int img_label = data -> labels[img_index]; 
        frequencies[img_label] += 1; // increment corresponding frequency by 1
    }
    
    *freq = -1; // set default values to get replaced
    *label = -1;
    // loop through frequencies and update correct values
    for (int k = 0; k < 10; k++) {
        if (frequencies[k] > *freq) { 
            *label = k;
            *freq = frequencies[k];
        } else if (frequencies[k] == *freq) {
            if (k < *label) { // store frequency of smaller label
                *label = k;
                *freq = frequencies[k];
            }
        } else {
            continue;
        }
    }
    
}

/**
 * Given a subset of M images as defined by their indices, find and return
 * the best pixel to split the data. The best pixel is the one which
 * has the minimum Gini impurity as computed by `gini_impurity()` and 
 * is not NAN. 
 * 
 * The return value will be a number between 0-783 (inclusive), representing
 *  the pixel the M images should be split based on.
 * 
 * If multiple pixels have the same minimal Gini impurity, return the smallest.
 */
int find_best_split(Dataset *data, int M, int *indices) {
    double min_impurity = INFINITY; 
    int best_split;

    // iterate through all pixels to find the minimum Gini impurity
    for (int i = 0; i < 784; i++) {
        double pixel_impurity = gini_impurity(data, M, indices, i);
        if (pixel_impurity < min_impurity && pixel_impurity != NAN) {
            best_split = i;
            min_impurity = pixel_impurity;
        } else if (pixel_impurity == min_impurity && pixel_impurity != NAN) {
            if (i < best_split) { // take the smaller pixel if Gini impurity is equal
                best_split = i;
                min_impurity = pixel_impurity;
            }
        } else {
            continue;
        }
    }

    return best_split;
}

/**
 * Helper function for build_subtree. 
 * Splits up the original `indices` array of length M based on whether pixel is less than 128. Updates 'left_size' 
 * and 'right_size' with new sizes of left and right subsets. Returns a nested array, where subsets[0] points to 
 * left node indices (Image indices with pixel value < 128) and subsets[1] points to right node indices 
 * (Image indices with pixel value >= 128).
 */
int **split_data(Dataset *data, int M, int *indices, int pixel, int *left_size, int *right_size) {
    // iterate through indices and increment size of left or right node array
    for (int i = 0; i < M; i++) {
        int index = indices[i];
        if (data->images[index].data[pixel] < 128) { // if pixel value < 128, increase size of left node arary.
            *left_size += 1;
        } else { // if pixel value >= 128, increase size of right node array.
            *right_size += 1;
        }
    }

    int **subsets = malloc(sizeof(int *) * 2); // nested array
    subsets[0] = malloc(sizeof(int) * (*left_size)); // points to array of pixels in left node subset
    subsets[1] = malloc(sizeof(int) * (*right_size)); // points to array of pixels in right node subset
    
    // populate subset arrays 
    int left_i = 0;
    int right_i = 0;
    for (int j = 0; j < M; j++) {
        int index = indices[j];
        if (data->images[index].data[pixel] < 128) { // if pixel value < 128, add Image index to left subset.
            subsets[0][left_i] = index;
            left_i += 1;
        } else { // if pixel value >= 128, add Image index to right subset.
            subsets[1][right_i] = index;
            right_i += 1;
        }
    }

    return subsets;
}

/**
 * Create the Decision tree. In each recursive call, consider the subset of the
 * dataset that correspond to the new node. To represent the subset, we pass 
 * an array of indices of these images in the subset of the dataset, along with 
 * its length M. 
 */
DTNode *build_subtree(Dataset *data, int M, int *indices) {
    // build new node
    DTNode *node = malloc(sizeof(DTNode));
    int *freq = malloc(sizeof(int));
    int *label = malloc(sizeof(int));
    get_most_frequent(data, M, indices, label, freq);

    if (( (double) *freq / (double) M) >= THRESHOLD_RATIO) { // create leaf node 
        node -> pixel = -1;
        node -> classification = *label;
        node -> left = NULL;
        node -> right = NULL;
    } else { // create node with left/right children
        int pixel_split = find_best_split(data, M, indices);
        node -> pixel = pixel_split;
        node -> classification = -1;
        // split data using helper function
        int *left_size = malloc(sizeof(int));
        int *right_size = malloc(sizeof(int));
        *left_size = 0;
        *right_size = 0;
        int **subsets = split_data(data, M, indices, pixel_split, left_size, right_size);
        // recurse on child nodes
        node -> left = build_subtree(data, *left_size, subsets[0]);
        node -> right = build_subtree(data, *right_size, subsets[1]);
        // free memory for subsets and int pointers
        free(left_size);
        free(right_size);
        free(subsets[0]);
        free(subsets[1]);
        free(subsets);
    }
    
    free(freq);
    free(label);

    return node;
}

/**
 * Function exposed to the user. Set up the `indices` array correctly for the 
 * entire dataset and call `build_subtree()`.
 */
DTNode *build_dec_tree(Dataset *data) {
    // set up 'indices' array
    int M = data -> num_items;
    int indices[M];
    for (int i = 0; i < M; i++) {
        indices[i] = i;
    }    

    // return the built tree
    return build_subtree(data, M, indices);
}

/**
 * Given a decision tree and an image to classify, return the predicted label.
 */
int dec_tree_classify(DTNode *root, Image *img) {
    if (root -> classification != -1) { // base case: if node is a leaf
        return root -> classification;
    } else {
        if ((img -> data)[root -> pixel] == 0) { // if pixel value is 0, recurse on left child
            return dec_tree_classify(root -> left, img);
        } else { // if pixel value is 255, recurse on right child
            return dec_tree_classify(root -> right, img);
        }
    }
}

/**
 * Free the decision tree.
 */
void free_dec_tree(DTNode *node) {
    if (node -> classification != -1) { // base case: free the leaf node
        free(node);
    } else { // recursive case: free children and then free parent node
        free_dec_tree(node -> left);
        free_dec_tree(node -> right);
        free(node);
    }
}

/**
 * Free all the allocated memory for the dataset.
 */
void free_dataset(Dataset *data) {
    // free array of pixel color values for each image
    for (int i = 0; i < data -> num_items; i++) {
        free((data -> images)[i].data);
    }
    
    // free images array
    free(data -> images);
    // free labels array
    free (data -> labels);
    // free dataset
    free(data);
}
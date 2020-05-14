# bigDataProject

Title: Restaurants classification using Apache Spark
# To do(demetris)

- [x] transform existing code to a function
- [x] make a main and call them from there
- [x] transform ratings 1,2,3->0 4,5->1

# To do(kyriakos)
- [x] make a function for each classifier 2/3
- [x] make a function to call them

# Function main()
-  if (file == train_set)
    - [x] parse the file and remove stopwords
    - [x] replace [1, 2, 3] to 0 and [4, 5] to 1
    - [x] classify with Logistic Regression
    - [x] classify with Navy Bayes
    - [ ] classify with Random Forest
    - [x] compare the result and select the most accuracy classifier
-  if (file == test_set)
    - [x] parse the file and remove stopwords
    - [x] classify with the best accuracy classifier

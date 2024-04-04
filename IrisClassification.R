# K Nearest Neighbors implementation

# Importing the dataset
data = iris

row_labels = data[,5]

# Encoding the target feature as factor
data$Species <- as.factor(data$Species)

# Scale the data since we will be using distance formulas on the data
# and we want to reduce complexity and computation when computing 
# especially when our datasets are huge!
data[,c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width")] <- scale(
  data[,c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width")])

# Split into test and train 80/20
set.seed(123)

size <- floor(0.8 *  nrow(data))

train_ind <- sample(seq_len(nrow(data)), size = size)

train_labels <- data[train_ind, 5]

data_train <- data[train_ind,1:4]
data_test <- data[-train_ind,1:4]

data_test_labels <- row_labels[-train_ind]

# Fit KNN Model
library(class)

# Find best k value
best_accuracy <- 0
best_k <- 0

for (k in 1:20) {
  predictions <- knn(train = data_train,
                     test = data_test,
                     cl = train_labels,
                     k = k)
  
  accuracy <- mean(predictions == data_test_labels)
  
  if (accuracy > best_accuracy) {
    best_accuracy <- accuracy
    best_k <- k
  }
}

cat("Best k value:", best_k, "\n")

# Fit KNN Model with best k value
predictions <- knn(train = data_train,
                   test = data_test,
                   cl = train_labels,
                   k = best_k)
library(ggplot2)
p1 <- ggplot(plot_predictions, aes(Petal.Length, Petal.Width, color = predicted, fill = predicted)) + 
  geom_point(size = 5) + 
  geom_text(aes(label=data_test_labels),hjust=1, vjust=2) +
  ggtitle("Predicted relationship between Petal Length and Width") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(legend.position = "none")

p2 <- ggplot(plot_predictions, aes(Sepal.Length, Sepal.Width, color = predicted, fill = predicted)) + 
  geom_point(size = 5) + 
  geom_text(aes(label=data_test_labels),hjust=1, vjust=2) +
  ggtitle("Predicted relationship between Sepal Length and Sepal") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(legend.position = "none")

# Model accuracy
accuracy <- mean(predictions == data_test_labels)
cat("Model accuracy:", accuracy, "\n")
print(p1)
print(p2)
# Confusion matrix
conf_matrix <- table(predictions, data_test_labels)
cat("Confusion Matrix:\n")
print(conf_matrix)

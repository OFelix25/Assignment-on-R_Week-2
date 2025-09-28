# Assignment on R
# wine_analysis analysis 

# Installing the different packages for analysis 

install.packages("dplyr")
install.packages("tidyr")
install.packages("caret")
install.packages("corrplot")
install.packages("readxl")
install.packages("writexl")
install.packages("factoextra")
install.packages("uwot")
install.packages("Rtsne")
install.packages("ggplot2")
install.packages("scales")
install.packages("gridExtra")
install.packages("RANN")

# Loading the different packages for analysis
library(dplyr)
library(tidyr)
library(caret)
library(corrplot)
library(readxl)
library(writexl)
library(factoextra)
library(uwot)
library(Rtsne)
library(ggplot2)
library(scales)
library(gridExtra)
library(RANN)

# Loading Excel files (Data was downloaded from Dataset:https://archive.ics.uci.edu/ml/datasets/wine+quality and saved in the a working folder)
red_wine <- read_excel("C:/Users/HP/Desktop/Felix Ochieng/CS/Felix Ochieng/Python/Data Analysis and Visualization/Assignment4/Ass4/winequality-red.xlsx")
white_wine <- read_excel("C:/Users/HP/Desktop/Felix Ochieng/CS/Felix Ochieng/Python/Data Analysis and Visualization/Assignment4/Ass4/winequality-white.xlsx")

# Adding the type column
red_wine$type <- "red"
white_wine$type <- "white"

# Combine the data for the two different types of wine (red and white) into one data frame
wine_all <- rbind(red_wine, white_wine)
write_xlsx(wine_all, "wine_all.xlsx")

# Data Cleaning process 
# Checking for missing values
colSums(is.na(wine_all))  # If any, impute with median
# wine_all <- wine_all %>% mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm=TRUE), .)))

# Removing the duplicates from the combined data
wine_all <- wine_all |> distinct()

# Ensuring  factors for categorical variables 
wine_all <- wine_all %>% mutate(across(where(is.character), as.factor))

# Outlier detection using IQR
num_cols <- wine_all %>% select(where(is.numeric))
iqr_outlier_flags <- num_cols %>%
  summarise(across(everything(), function(x) {
    Q1 <- quantile(x, 0.25)
    Q3 <- quantile(x, 0.75)
    IQRv <- Q3 - Q1
    sum(x < (Q1 - 1.5*IQRv) | x > (Q3 + 1.5*IQRv))
  })) %>% pivot_longer(cols = everything(), names_to = "feature", values_to = "outlier_count")
print(iqr_outlier_flags)

# Scaling numeric features (zero mean, unit variance)
num_vars <- names(num_cols)
preProc <- preProcess(wine_all %>% select(all_of(num_vars)), method = c("center","scale"))
X_scaled <- predict(preProc, wine_all %>% select(all_of(num_vars)))

# Re-integrating the categorical variables back into the combined data set
X_scaled$type <- wine_all$type
X_scaled$quality <- wine_all$quality

# Feature Analysis
# Low variance features
nzv <- nearZeroVar(X_scaled %>% select(-type,-quality), saveMetrics = TRUE)
print(nzv[nzv$nzv==TRUE,])

# Correlation reduction
cor_mat <- cor(X_scaled %>% select(-type,-quality))
corrplot(cor_mat, method="color", tl.cex=0.8, number.cex=0.6)
highCorr <- findCorrelation(cor_mat, cutoff=0.9, names=TRUE)
reduced_vars <- setdiff(colnames(X_scaled %>% select(-type,-quality)), highCorr)
X_final <- X_scaled %>% select(all_of(reduced_vars))
X_final_mat <- as.matrix(X_final)

# Dimensionality Reduction

set.seed(123)

# PCA
pca_res <- prcomp(X_final_mat, center=FALSE, scale.=FALSE)
pca_df <- as.data.frame(pca_res$x[,1:2])
pca_df$quality <- factor(wine_all$quality)
pca_df$type <- wine_all$type

# UMAP
umap_res <- umap(X_final_mat, n_neighbors=15, n_components=2, metric="euclidean")
umap_df <- as.data.frame(umap_res)
colnames(umap_df) <- c("UMAP1","UMAP2")
umap_df$quality <- factor(wine_all$quality)
umap_df$type <- wine_all$type

# t-SNE
tsne_res <- Rtsne(X_final_mat, dims=2, perplexity=30, check_duplicates=FALSE)
tsne_df <- as.data.frame(tsne_res$Y)
colnames(tsne_df) <- c("TSNE1","TSNE2")
tsne_df$quality <- factor(wine_all$quality)
tsne_df$type <- wine_all$type

# Visualization
p1 <- ggplot(pca_df, aes(x=PC1, y=PC2, color=quality, shape=type)) + geom_point(alpha=0.6) + labs(title="PCA - Wine Quality")
p2 <- ggplot(umap_df, aes(x=UMAP1, y=UMAP2, color=quality, shape=type)) + geom_point(alpha=0.6) + labs(title="UMAP - Wine Quality")
p3 <- ggplot(tsne_df, aes(x=TSNE1, y=TSNE2, color=quality, shape=type)) + geom_point(alpha=0.6) + labs(title="t-SNE - Wine Quality")

grid.arrange(p1,p2,p3,nrow=2)

# Saving the plots
ggsave("PCA_plot.png", p1, width=8, height=6)
ggsave("UMAP_plot.png", p2, width=8, height=6)
ggsave("TSNE_plot.png", p3, width=8, height=6)

# k-NN preservation
preservation <- function(original_mat, embedded_mat, k=10){
  orig_nn <- nn2(data=original_mat,k=k+1)$nn.idx[,-1]
  emb_nn <- nn2(data=as.matrix(embedded_mat), k=k+1)$nn.idx[,-1]
  preserved <- sapply(1:nrow(original_mat), function(i) length(intersect(orig_nn[i,], emb_nn[i,]))/k)
  mean(preserved)
}

knn_pres <- tibble(
  method=c("PCA","UMAP","tSNE"),
  knn_preservation=c(
    preservation(X_final_mat, pca_res$x[,1:2]),
    preservation(X_final_mat, umap_res),
    preservation(X_final_mat, tsne_res$Y)
  )
)
print(knn_pres)

# Reflection
cat("Top features by absolute loadings on PC1:\n")
pc1_loadings <- sort(abs(pca_res$rotation[,1]), decreasing=TRUE)
print(head(pc1_loadings,6))

cat("\nMean values by wine type:\n")
type_means <- wine_all %>% group_by(type) %>% summarise(across(where(is.numeric), mean))
print(type_means)

# Saving reduced datasets
write.csv(pca_df, "wine_pca2.csv", row.names=FALSE)
write.csv(umap_df, "wine_umap2.csv", row.names=FALSE)
write.csv(tsne_df, "wine_tsne2.csv", row.names=FALSE)
write.csv(knn_pres, "knn_preservation.csv", row.names=FALSE)

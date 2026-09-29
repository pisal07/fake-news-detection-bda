# Fake News Detection - Big Data Analytics Project
# Models: Naive Bayes, Random Forest, SVM

setwd("C:/Users/admin/Downloads/bda")
graphics.off()
rm(list = ls())

library(tm)
library(SnowballC)
library(wordcloud)
library(caret)
library(e1071)
library(randomForest)
library(ggplot2)
library(pROC)
library(reshape2)
library(RColorBrewer)

start_time <- Sys.time()

# 1. Dataset Ingestion & Standardization
standardize_cols <- function(df, default_label = "FAKE") {
  names(df) <- tolower(names(df))
  
  t_col <- intersect(names(df), c("title", "headline", "headlines", "heading"))[1]
  b_col <- intersect(names(df), c("text", "article", "body", "content", "news"))[1]
  l_col <- intersect(names(df), c("label", "class", "target", "category", "verdict"))[1]
  
  df$title <- if (!is.na(t_col)) df[[t_col]] else ""
  df$text  <- if (!is.na(b_col)) df[[b_col]] else ""
  df$label <- if (!is.na(l_col)) df[[l_col]] else default_label
  
  if (is.numeric(df$label)) {
    df$label <- ifelse(df$label == 0, "FAKE", "REAL")
  } else {
    df$label <- toupper(as.character(df$label))
    df$label <- ifelse(grepl("FAKE|FALSE", df$label), "FAKE", "REAL")
  }
  return(df[, c("title", "text", "label")])
}

cat("Loading datasets...\n")
news <- rbind(
  standardize_cols(read.csv("Fake.csv", stringsAsFactors = FALSE), "FAKE"),
  standardize_cols(read.csv("True.csv", stringsAsFactors = FALSE), "REAL"),
  standardize_cols(read.csv("combined_output.csv", stringsAsFactors = FALSE)),
  standardize_cols(read.csv("dataset2_welfake_clean.csv", stringsAsFactors = FALSE))
)

news$text <- paste(news$title, news$text)
news$label <- as.factor(news$label)

# 2. Text Preprocessing & Cleaning
news$text <- gsub("\\(reuters\\)|reuters", "", news$text, ignore.case = TRUE)
news$text <- gsub("^[A-Z ]+\\s*-\\s*", "", news$text)

# Subsampling 5000 records for stable memory convergence
set.seed(42)
if (nrow(news) > 5000) {
  news <- news[sample(nrow(news), 5000), ]
}

# 3. Stylometric Feature Extraction
news$excl_count <- sapply(news$text, function(x) length(regmatches(x, gregexpr("!", x))[[1]]))
news$caps_count <- sapply(news$text, function(x) length(regmatches(x, gregexpr("\\b[A-Z]{2,}\\b", x))[[1]]))
news$word_count <- sapply(gregexpr("\\W+", news$text), length) + 1

# 4. Document-Term Matrix (TF)
cat("Building Document Term Matrix...\n")
corpus <- VCorpus(VectorSource(news$text))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, stemDocument)

dtm <- DocumentTermMatrix(corpus, control = list(weighting = weightTf))
dtm <- removeSparseTerms(dtm, 0.98)
dtm_df <- as.data.frame(as.matrix(dtm))
colnames(dtm_df) <- make.names(colnames(dtm_df))

dtm_df$excl_count <- news$excl_count
dtm_df$caps_count <- news$caps_count
dtm_df$word_count <- news$word_count
dtm_df$label      <- news$label

# 5. Train / Test Split (80 / 20)
set.seed(42)
split_idx  <- createDataPartition(dtm_df$label, p = 0.8, list = FALSE)
train_data <- dtm_df[split_idx, ]
test_data  <- dtm_df[-split_idx, ]

get_metrics <- function(pred, actual, scores = NULL) {
  cm <- confusionMatrix(pred, actual, positive = "FAKE")
  auc_val <- NA
  if (!is.null(scores)) {
    r <- roc(actual, scores, levels = c("REAL", "FAKE"), quiet = TRUE)
    if (auc(r) < 0.5) r <- roc(actual, -scores, levels = c("REAL", "FAKE"), quiet = TRUE)
    auc_val <- as.numeric(auc(r))
  }
  list(
    cm = cm,
    Accuracy  = as.numeric(cm$overall["Accuracy"]),
    Precision = as.numeric(cm$byClass["Precision"]),
    Recall    = as.numeric(cm$byClass["Recall"]),
    F1        = as.numeric(cm$byClass["F1"]),
    AUC       = auc_val
  )
}

print_summary <- function(name, res) {
  cat("\n=========================================\n")
  cat("  MODEL:", name, "\n")
  cat("=========================================\n")
  print(res$cm$table)
  cat(sprintf("\nAccuracy : %.4f (%.2f%%)", res$Accuracy, res$Accuracy * 100))
  cat(sprintf("\nPrecision: %.4f (%.2f%%)", res$Precision, res$Precision * 100))
  cat(sprintf("\nRecall   : %.4f (%.2f%%)", res$Recall, res$Recall * 100))
  cat(sprintf("\nF1-Score : %.4f", res$F1))
  cat(sprintf("\nAUC      : %.4f\n", res$AUC))
  cat("-----------------------------------------\n\n")
}

# 6. Model Training & Evaluation
cat("Training Naive Bayes...\n")
nb_model <- naiveBayes(label ~ ., data = train_data, laplace = 1)
nb_pred  <- predict(nb_model, test_data)
nb_prob  <- predict(nb_model, test_data, type = "raw")[, "FAKE"]
nb_res   <- get_metrics(nb_pred, test_data$label, nb_prob)
print_summary("NAIVE BAYES", nb_res)

cat("Training Random Forest...\n")
rf_model <- randomForest(label ~ ., data = train_data, ntree = 100, importance = TRUE)
rf_pred  <- predict(rf_model, test_data)
rf_prob  <- predict(rf_model, test_data, type = "prob")[, "FAKE"]
rf_res   <- get_metrics(rf_pred, test_data$label, rf_prob)
print_summary("RANDOM FOREST", rf_res)

cat("Training Support Vector Machine...\n")
svm_model <- svm(label ~ ., data = train_data, kernel = "linear", scale = TRUE, cost = 1)
svm_pred  <- predict(svm_model, test_data, decision.values = TRUE)
svm_dv    <- as.numeric(attr(svm_pred, "decision.values"))
svm_res   <- get_metrics(svm_pred, test_data$label, svm_dv)
print_summary("SUPPORT VECTOR MACHINE", svm_res)

# 7. Save Artifacts for Shiny
saveRDS(rf_model, "rf_model.rds")
saveRDS(colnames(train_data)[colnames(train_data) != "label"], "features.rds")

# 8. Comparison Results & CSV Export
results <- data.frame(
  Model     = c("Naive Bayes", "Random Forest", "Support Vector Machine"),
  Accuracy  = c(nb_res$Accuracy,  rf_res$Accuracy,  svm_res$Accuracy),
  Precision = c(nb_res$Precision, rf_res$Precision, svm_res$Precision),
  Recall    = c(nb_res$Recall,    rf_res$Recall,    svm_res$Recall),
  F1_Score  = c(nb_res$F1,        rf_res$F1,        svm_res$F1),
  AUC       = c(nb_res$AUC,       rf_res$AUC,       svm_res$AUC)
)
write.csv(results, "model_comparison_table.csv", row.names = FALSE)

cat("\nFinal Comparison Summary:\n")
print(results)

# 9. Plots & Visualizations
results_long <- melt(results, id.vars = "Model", variable.name = "Metric", value.name = "Score")
p_comp <- ggplot(results_long, aes(x = Metric, y = Score, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7) +
  geom_text(aes(label = round(Score, 2)), position = position_dodge(0.8), vjust = -0.4, size = 3) +
  ylim(0, 1.1) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Model Performance Benchmark", x = "", y = "Score") +
  theme_minimal()
ggsave("model_comparison_all_metrics.png", plot = p_comp, width = 8, height = 5)

# ROC Curves
roc_nb  <- roc(test_data$label, nb_prob, levels = c("REAL", "FAKE"), quiet = TRUE)
roc_rf  <- roc(test_data$label, rf_prob, levels = c("REAL", "FAKE"), quiet = TRUE)
roc_svm <- roc(test_data$label, svm_dv, levels = c("REAL", "FAKE"), quiet = TRUE)
if (auc(roc_svm) < 0.5) roc_svm <- roc(test_data$label, -svm_dv, levels = c("REAL", "FAKE"), quiet = TRUE)

png("roc_curves_comparison.png", width = 700, height = 600)
plot(roc_rf, col = "#2c7bb6", lwd = 2.5, main = "ROC Curves Comparison")
lines(roc_svm, col = "#fdae61", lwd = 2)
lines(roc_nb, col = "#d7191c", lwd = 2, lty = 2)
legend("bottomright",
       legend = c(paste0("Random Forest (AUC = ", round(auc(roc_rf), 3), ")"),
                  paste0("SVM (AUC = ", round(auc(roc_svm), 3), ")"),
                  paste0("Naive Bayes (AUC = ", round(auc(roc_nb), 3), ")")),
       col = c("#2c7bb6", "#fdae61", "#d7191c"), lwd = c(2.5, 2, 2), lty = c(1, 1, 2))
grid()
dev.off()

# Feature Importance Plot
png("feature_importance_rf.png", width = 750, height = 750)
varImpPlot(rf_model, n.var = 20, main = "Top 20 Features - Random Forest")
dev.off()

# Confusion Matrices
plot_cm <- function(cm_obj, title_text, filename, fill_color) {
  df <- as.data.frame(cm_obj$table)
  colnames(df) <- c("Prediction", "Reference", "Freq")
  p <- ggplot(df, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), size = 5, fontface = "bold") +
    scale_fill_gradient(low = "#f7fbff", high = fill_color) +
    labs(title = title_text) +
    theme_minimal()
  ggsave(filename, plot = p, width = 4.8, height = 3.8)
}

plot_cm(nb_res$cm,  "Confusion Matrix - Naive Bayes",   "confusion_matrix_nb.png",  "#fc9272")
plot_cm(rf_res$cm,  "Confusion Matrix - Random Forest", "confusion_matrix_rf.png",  "#9ecae1")
plot_cm(svm_res$cm, "Confusion Matrix - SVM",           "confusion_matrix_svm.png", "#a1d99b")

# Word Clouds
save_cloud <- function(text_vec, filename, pal) {
  words <- unlist(strsplit(tolower(paste(sample(text_vec, min(500, length(text_vec))), collapse = " ")), "\\W+"))
  words <- words[nchar(words) > 3 & !(words %in% stopwords("english"))]
  w_counts <- sort(table(words), decreasing = TRUE)
  
  png(filename, width = 700, height = 700)
  par(mar = c(0, 0, 0, 0))
  wordcloud(names(w_counts), as.numeric(w_counts), max.words = 75,
            colors = pal, scale = c(3.5, 0.7), random.order = FALSE)
  dev.off()
}

save_cloud(news$text[news$label == "FAKE"], "wordcloud_fake.png", brewer.pal(8, "Dark2"))
save_cloud(news$text[news$label == "REAL"], "wordcloud_real.png", brewer.pal(8, "Blues")[3:8])

graphics.off()
total_time <- round(difftime(Sys.time(), start_time, units = "mins"), 2)
cat("\nExecution complete in", total_time, "minutes.\n")
